import os

import logging
import arviz as az
import bambi as bmb
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from pysal.lib import weights
from shapely import wkt
from shapely.geometry import Polygon
from spreg import Panel_FE_Lag, dgp_lag

from .data_pull import DataPull


class SpatialReg(DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
        seed: int = 787,
    ):
        super().__init__(saving_dir, database_file, log_file)
        self.seed = seed

    def spatial_data(
        self, mu: int, sigma: int, rho: float, time: int
    ) -> gpd.GeoDataFrame:
        rng_global = np.random.default_rng(787)
        # Number of observations
        gdf = self.spatial_df()
        n_obs = len(gdf)

        # Create the independent variables for 4 variables that com from a normal distribution X ~ N(mu, sigma)
        X = np.ones((n_obs, 4))

        for i in range(1, 4):
            rng = np.random.default_rng(seed=self.seed + 1)
            X[:, i] = rng.normal(loc=mu, scale=sigma, size=n_obs)

        # Define Beta coefficients
        beta = np.array([4, 5, 6, 7])

        # Compute XB matrix
        xb = X @ beta
        xb = xb.reshape(-1, 1)

        u = rng_global.normal(loc=0, scale=2, size=n_obs).reshape(-1, 1)

        # Define the spatial weight matrix
        wr = weights.contiguity.Rook.from_dataframe(gdf, use_index=False)
        wr = wr.transform = "r"

        wq = weights.contiguity.Queen.from_dataframe(gdf, use_index=False)
        wq.transform = "r"

        wk6 = weights.KNN.from_dataframe(gdf, k=6, use_index=False)
        wk6.transform = "r"

        # calculate the spatial lag
        y_d = dgp_lag(u, xb, wq, rho=rho)
        gdf["y_d"] = y_d

        # pre calculate the spatial lag and apply them
        gdf["X_1"] = X[:, 1]
        gdf["X_2"] = X[:, 2]
        gdf["X_3"] = X[:, 3]
        gdf["w_rook"] = weights.lag_spatial(wr, y_d)
        gdf["w_queen"] = weights.lag_spatial(wq, y_d)
        gdf["w_knn6"] = weights.lag_spatial(wk6, y_d)
        gdf["time"] = time

        return gdf

    def spatial_panel(self, time, rho):
        gdf = gpd.GeoDataFrame(
            [
                "zipcode",
                "geometry",
                "y_d",
                "X_1",
                "X_2",
                "X_3",
                "w_rook",
                "w_queen",
                "w_knn6",
                "time",
            ]
        )
        for time_period in range(0, time):
            # Remove columns with all NA values from gdf and tmp
            tmp = self.spatial_data(mu=2, sigma=3, rho=rho, time=time_period)
            # gdf = gdf.dropna(axis=1, how='all')
            # tmp = tmp.dropna(axis=1, how='all')

            gdf = pd.concat([gdf, tmp]).reset_index(drop=True)

        return gdf

    def spatial_simulation(self, time, rho, simulations):
        df = pl.DataFrame(
            [
                pl.Series("bayes_x", [], dtype=pl.Float64),
                pl.Series("bayes_w", [], dtype=pl.Float64),
                pl.Series("freq_x", [], dtype=pl.Float64),
                pl.Series("freq_w", [], dtype=pl.Float64),
                pl.Series("simulations_id", [], dtype=pl.Int32),
            ]
        )
        for i in range(0, simulations):
            gdf = self.spatial_panel(time=time, rho=rho)
            wr = weights.contiguity.Rook.from_dataframe(gdf[gdf["time"] == 0])
            wr.transform = "r"

            y_d = gdf["y_d"].values.reshape(-1, 1)
            xb = gdf["X1"].values.reshape(-1, 1)
            fe_lag = Panel_FE_Lag(y=y_d, x=xb, w=wr)

            gdf = gdf.drop("geometry", axis=1)
            model = bmb.Model("y_d ~ 1 + X1 + w_d", gdf, dropna=True)
            results = model.fit(cores=10)
            az_summary = az.summary(
                results, hdi_prob=0.95
            )  # You can adjust hdi_prob if needed
            means = az_summary["mean"]

            df_sim = pl.DataFrame(
                [
                    pl.Series("bayes_x", [means.iloc[1]], dtype=pl.Float64),
                    pl.Series("bayes_w", [means.iloc[3]], dtype=pl.Float64),
                    pl.Series("freq_x", [fe_lag.betas[0][0]], dtype=pl.Float64),
                    pl.Series("freq_w", [fe_lag.betas[1][0]], dtype=pl.Float64),
                    pl.Series("simulations_id", [i], dtype=pl.Int32),
                ]
            )
            df = pl.concat([df, df_sim], how="vertical")
            logging.info(f"Completed Simulation #{i} succesfully")

        self.results = {
            "bayes_x": (
                df.select((pl.col("bayes_x") - rho) ** 2).sum() / simulations
            ).item(),
            "bayes_w": (
                df.select((pl.col("bayes_w") - rho) ** 2).sum() / simulations
            ).item(),
            "freq_x": (
                df.select((pl.col("freq_x") - rho) ** 2).sum() / simulations
            ).item(),
            "freq_w": (
                df.select((pl.col("freq_w") - rho) ** 2).sum() / simulations
            ).item(),
        }
        return df

    def spatial_df(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(self.make_spatial_table())
        gdf["geometry"] = gdf["geometry"].apply(wkt.loads)
        gdf = gdf.set_geometry("geometry").set_crs("EPSG:4269", allow_override=True)
        gdf = gdf.to_crs("EPSG:3395")
        gdf["zipcode"] = gdf["zipcode"].astype(str)
        return gdf

    def calculate_spatial_lag(self, df, w, column):
        # Reshape y to match the number of rows in the dataframe
        y = df[column].values.reshape(-1, 1)

        # Apply spatial lag
        spatial_lag = weights.lag_spatial(w, y)

        return spatial_lag
