import logging
import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from pysal.lib import weights
from shapely import wkt
from spreg import dgp_lag
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from patsy import dmatrix
import pymc as pm

from .data_pull import DataPull


class SpatialReg(DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

        # Define the spatial weight matrix
        self.wr = weights.contiguity.Rook.from_dataframe(
            self.spatial_df(), use_index=False
        )

        self.wq = weights.contiguity.Queen.from_dataframe(
            self.spatial_df(), use_index=False
        )
        self.wq.transform = "r"

        self.wk6 = weights.KNN.from_dataframe(self.spatial_df(), k=6, use_index=False)
        self.wk6.transform = "r"

    def spatial_data(
        self, mu: int, sigma: int, rho: float, time: int, seed: int
    ) -> gpd.GeoDataFrame:
        rng_global = np.random.default_rng(seed=seed)
        # Number of observations
        gdf = self.spatial_df()
        n_obs = len(gdf)

        # Create the independent variables for 4 variables that com from a normal distribution X ~ N(mu, sigma)
        X = np.ones((n_obs, 4))

        for i in range(1, 4):
            rng = np.random.default_rng(seed=seed + i)
            X[:, i] = rng.normal(loc=mu, scale=sigma, size=n_obs)

        # Define Beta coefficients
        beta = np.array([4, 5, 6, 7])

        # Compute XB matrix
        xb = X @ beta
        xb = xb.reshape(-1, 1)

        u = rng_global.normal(loc=0, scale=2, size=n_obs).reshape(-1, 1)

        # calculate the spatial lag
        y_true = dgp_lag(u, xb, self.wq, rho=rho)

        gdf["y_true"] = y_true

        # pre calculate the spatial lag and apply them
        gdf["X_1"] = X[:, 1]
        gdf["X_2"] = X[:, 2]
        gdf["X_3"] = X[:, 3]
        gdf["centroid"] = gdf.centroid
        gdf["lat"] = gdf["centroid"].x
        gdf["lon"] = gdf["centroid"].y
        gdf["w_rook"] = weights.lag_spatial(self.wr, y_true)
        gdf["w_queen"] = weights.lag_spatial(self.wq, y_true)
        gdf["w_knn6"] = weights.lag_spatial(self.wk6, y_true)
        gdf["time"] = time

        return gdf

    def spatial_panel(self, time, rho, seed):
        gdf = gpd.GeoDataFrame(
            columns=[
                "zipcode",
                "geometry",
                "y_true",
                "X_1",
                "X_2",
                "X_3",
                "centroid",
                "lat",
                "lon",
                "w_rook",
                "w_queen",
                "w_knn6",
                "time",
            ]
        )
        for time_period in range(0, time):
            # Remove columns with all NA values from gdf and tmp
            tmp = self.spatial_data(mu=2, sigma=3, rho=rho, time=time_period, seed=seed)
            tmp = tmp.dropna(axis=1, how="all")

            gdf = pd.concat([gdf, tmp]).reset_index(drop=True)

        return gdf

    def spatial_simulation(self, time, rho, simulations, start_seed):
        logging.getLogger("pymc").setLevel(logging.WARNING)
        df_simulation = pl.DataFrame(
            [
                # pl.Series("bayes_rook_X1", [], dtype=pl.Float64),
                # pl.Series("bayes_rook_X2", [], dtype=pl.Float64),
                # pl.Series("bayes_rook_X3", [], dtype=pl.Float64),
                # pl.Series("bayes_rook_rho", [], dtype=pl.Float64),
                # pl.Series("bayes_rook_intercept", [], dtype=pl.Float64),
                # pl.Series("bayes_queen_X1", [], dtype=pl.Float64),
                # pl.Series("bayes_queen_X2", [], dtype=pl.Float64),
                # pl.Series("bayes_queen_X3", [], dtype=pl.Float64),
                # pl.Series("bayes_queen_rho", [], dtype=pl.Float64),
                # pl.Series("bayes_queen_intercept", [], dtype=pl.Float64),
                # pl.Series("bayes_knn6_X1", [], dtype=pl.Float64),
                # pl.Series("bayes_knn6_X2", [], dtype=pl.Float64),
                # pl.Series("bayes_knn6_X3", [], dtype=pl.Float64),
                # pl.Series("bayes_knn6_rho", [], dtype=pl.Float64),
                # pl.Series("bayes_knn6_intercept", [], dtype=pl.Float64),
                # pl.Series("bayes_base_X1", [], dtype=pl.Float64),
                # pl.Series("bayes_base_X2", [], dtype=pl.Float64),
                # pl.Series("bayes_base_X3", [], dtype=pl.Float64),
                # pl.Series("bayes_base_intercept", [], dtype=pl.Float64),
                pl.Series("freq_rook_X1", [], dtype=pl.Float64),
                pl.Series("freq_rook_X2", [], dtype=pl.Float64),
                pl.Series("freq_rook_X3", [], dtype=pl.Float64),
                pl.Series("freq_rook_rho", [], dtype=pl.Float64),
                pl.Series("freq_rook_intercept", [], dtype=pl.Float64),
                pl.Series("freq_queen_X1", [], dtype=pl.Float64),
                pl.Series("freq_queen_X2", [], dtype=pl.Float64),
                pl.Series("freq_queen_X3", [], dtype=pl.Float64),
                pl.Series("freq_queen_rho", [], dtype=pl.Float64),
                pl.Series("freq_queen_intercept", [], dtype=pl.Float64),
                pl.Series("freq_knn6_X1", [], dtype=pl.Float64),
                pl.Series("freq_knn6_X2", [], dtype=pl.Float64),
                pl.Series("freq_knn6_X3", [], dtype=pl.Float64),
                pl.Series("freq_knn6_rho", [], dtype=pl.Float64),
                pl.Series("freq_knn6_intercept", [], dtype=pl.Float64),
                pl.Series("freq_base_X1", [], dtype=pl.Float64),
                pl.Series("freq_base_X2", [], dtype=pl.Float64),
                pl.Series("freq_base_X3", [], dtype=pl.Float64),
                pl.Series("freq_base_intercept", [], dtype=pl.Float64),
                pl.Series("freq_tensor_X1", [], dtype=pl.Float64),
                pl.Series("freq_tensor_X2", [], dtype=pl.Float64),
                pl.Series("freq_tensor_X3", [], dtype=pl.Float64),
                pl.Series("freq_tensor_intercept", [], dtype=pl.Float64),
                pl.Series("simulknations_id", [], dtype=pl.Int32),
            ]
        )

        for i in range(simulations):
            start_seed += 1
            gdf = self.spatial_panel(time=time, rho=rho, seed=start_seed)
            df = gdf.drop("geometry", axis=1)

            # # Bayesian regressions
            # results_rook = self.bayes_reg(data=df, weight="w_rook")
            # az_rook = az.summary(results_rook, hdi_prob=0.95)
            # bayes_rook = az_rook["mean"]

            # results_queen = self.bayes_reg(data=df, weight="w_queen")
            # az_queen = az.summary(results_queen, hdi_prob=0.95)
            # bayes_queen = az_queen["mean"]

            # results_knn6 = self.bayes_reg(data=df, weight="w_knn6")
            # az_knn6 = az.summary(results_knn6, hdi_prob=0.95)
            # bayes_knn6 = az_knn6["mean"]

            # # Bayesian base (no spatial weight)
            # y_true = df["y_true"].values
            # X_1 = df["X_1"].values
            # X_2 = df["X_2"].values
            # X_3 = df["X_3"].values

            # with pm.Model() as model:
            #     sigma = pm.HalfCauchy("sigma", beta=10)
            #     intercept = pm.Normal("intercept", 0, sigma=20)
            #     beta_1 = pm.Normal("X_1", 0, sigma=10)
            #     beta_2 = pm.Normal("X_2", 0, sigma=10)
            #     beta_3 = pm.Normal("X_3", 0, sigma=10)

            #     pm.Normal(
            #         "y_true",
            #         mu=intercept + beta_1 * X_1 + beta_2 * X_2 + beta_3 * X_3,
            #         sigma=sigma,
            #         observed=y_true,
            #     )

            #     idata = pm.sample(chains=10, progressbar=False)

            # az_base = az.summary(idata, hdi_prob=0.95)
            # bayes_base = az_base["mean"]

            # Tensor regression smoothing
            formula = "X_1 + X_2 + X_3 + te(cr(lat, df=6), cr(lon, df=6), constraints='center')"
            design = dmatrix(formula, gdf)

            model = Ridge(alpha=1e-3)
            model.fit(design, gdf["y_true"])
            column_names = design.design_info.column_names

            coef = model.coef_
            coef_dict = dict(zip(column_names, coef))

            # Frequentist regressions
            results_rook = self.freq_reg(data=df, weights="w_rook")
            results_queen = self.freq_reg(data=df, weights="w_queen")
            results_knn6 = self.freq_reg(data=df, weights="w_knn6")
            results_base = self.freq_reg(data=df, weights="")

            df_sim = pl.DataFrame(
                [
                    # pl.Series("bayes_rook_X1", [bayes_rook["X_1"]]),
                    # pl.Series("bayes_rook_X2", [bayes_rook["X_2"]]),
                    # pl.Series("bayes_rook_X3", [bayes_rook["X_3"]]),
                    # pl.Series("bayes_rook_rho", [bayes_rook["rho"]]),
                    # pl.Series("bayes_rook_intercept", [bayes_rook["intercept"]]),
                    # pl.Series("bayes_queen_X1", [bayes_queen["X_1"]]),
                    # pl.Series("bayes_queen_X2", [bayes_queen["X_2"]]),
                    # pl.Series("bayes_queen_X3", [bayes_queen["X_3"]]),
                    # pl.Series("bayes_queen_rho", [bayes_queen["rho"]]),
                    # pl.Series("bayes_queen_intercept", [bayes_queen["intercept"]]),
                    # pl.Series("bayes_knn6_X1", [bayes_knn6["X_1"]]),
                    # pl.Series("bayes_knn6_X2", [bayes_knn6["X_2"]]),
                    # pl.Series("bayes_knn6_X3", [bayes_knn6["X_3"]]),
                    # pl.Series("bayes_knn6_rho", [bayes_knn6["rho"]]),
                    # pl.Series("bayes_knn6_intercept", [bayes_knn6["intercept"]]),
                    # pl.Series("bayes_base_X1", [bayes_base["X_1"]]),
                    # pl.Series("bayes_base_X2", [bayes_base["X_2"]]),
                    # pl.Series("bayes_base_X3", [bayes_base["X_3"]]),
                    # pl.Series("bayes_base_intercept", [bayes_base["intercept"]]),
                    pl.Series("freq_rook_X1", [results_rook.params[1]]),
                    pl.Series("freq_rook_X2", [results_rook.params[2]]),
                    pl.Series("freq_rook_X3", [results_rook.params[3]]),
                    pl.Series("freq_rook_rho", [results_rook.params[4]]),
                    pl.Series("freq_rook_intercept", [results_rook.params[0]]),
                    pl.Series("freq_queen_X1", [results_queen.params[1]]),
                    pl.Series("freq_queen_X2", [results_queen.params[2]]),
                    pl.Series("freq_queen_X3", [results_queen.params[3]]),
                    pl.Series("freq_queen_rho", [results_queen.params[4]]),
                    pl.Series("freq_queen_intercept", [results_queen.params[0]]),
                    pl.Series("freq_knn6_X1", [results_knn6.params[1]]),
                    pl.Series("freq_knn6_X2", [results_knn6.params[2]]),
                    pl.Series("freq_knn6_X3", [results_knn6.params[3]]),
                    pl.Series("freq_knn6_rho", [results_knn6.params[4]]),
                    pl.Series("freq_knn6_intercept", [results_knn6.params[0]]),
                    pl.Series("freq_base_X1", [results_base.params[1]]),
                    pl.Series("freq_base_X2", [results_base.params[2]]),
                    pl.Series("freq_base_X3", [results_base.params[3]]),
                    pl.Series("freq_base_intercept", [results_base.params[0]]),
                    pl.Series("freq_tensor_X1", [coef_dict["X_1"]]),
                    pl.Series("freq_tensor_X2", [coef_dict["X_2"]]),
                    pl.Series("freq_tensor_X3", [coef_dict["X_3"]]),
                    pl.Series("freq_tensor_intercept", [model.intercept_]),
                    pl.Series("simulknations_id", [i], dtype=pl.Int32),
                ]
            )

            df_simulation = pl.concat([df_simulation, df_sim], how="vertical")
            logging.info(f"Completed Simulation #{i} successfully")

            self.results = {
                # # Bayes Rook
                # "bayes_rook_intercept": (
                #     df_simulation.select(
                #         (pl.col("bayes_rook_intercept") - 4) ** 2
                #     ).sum()
                #     / simulations
                # ).item(),
                # "bayes_rook_X1": (
                #     df_simulation.select((pl.col("bayes_rook_X1") - 5) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_rook_X2": (
                #     df_simulation.select((pl.col("bayes_rook_X2") - 6) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_rook_X3": (
                #     df_simulation.select((pl.col("bayes_rook_X3") - 7) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_rook_rho": (
                #     df_simulation.select((pl.col("bayes_rook_rho") - rho) ** 2).sum()
                #     / simulations
                # ).item(),
                # # Bayes Queen
                # "bayes_queen_intercept": (
                #     df_simulation.select(
                #         (pl.col("bayes_queen_intercept") - 4) ** 2
                #     ).sum()
                #     / simulations
                # ).item(),
                # "bayes_queen_X1": (
                #     df_simulation.select((pl.col("bayes_queen_X1") - 5) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_queen_X2": (
                #     df_simulation.select((pl.col("bayes_queen_X2") - 6) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_queen_X3": (
                #     df_simulation.select((pl.col("bayes_queen_X3") - 7) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_queen_rho": (
                #     df_simulation.select((pl.col("bayes_queen_rho") - rho) ** 2).sum()
                #     / simulations
                # ).item(),
                # # Bayes KNN6
                # "bayes_knn6_intercept": (
                #     df_simulation.select(
                #         (pl.col("bayes_knn6_intercept") - 4) ** 2
                #     ).sum()
                #     / simulations
                # ).item(),
                # "bayes_knn6_X1": (
                #     df_simulation.select((pl.col("bayes_knn6_X1") - 5) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_knn6_X2": (
                #     df_simulation.select((pl.col("bayes_knn6_X2") - 6) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_knn6_X3": (
                #     df_simulation.select((pl.col("bayes_knn6_X3") - 7) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_knn6_rho": (
                #     df_simulation.select((pl.col("bayes_knn6_rho") - rho) ** 2).sum()
                #     / simulations
                # ).item(),
                # # Bayes Base (no spatial weight)
                # "bayes_base_intercept": (
                #     df_simulation.select(
                #         (pl.col("bayes_base_intercept") - 4) ** 2
                #     ).sum()
                #     / simulations
                # ).item(),
                # "bayes_base_X1": (
                #     df_simulation.select((pl.col("bayes_base_X1") - 5) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_base_X2": (
                #     df_simulation.select((pl.col("bayes_base_X2") - 6) ** 2).sum()
                #     / simulations
                # ).item(),
                # "bayes_base_X3": (
                #     df_simulation.select((pl.col("bayes_base_X3") - 7) ** 2).sum()
                #     / simulations
                # ).item(),
                # Frequency Rook
                "freq_rook_intercept": (
                    df_simulation.select((pl.col("freq_rook_intercept") - 1) ** 2).sum()
                    / simulations
                ).item(),
                "freq_rook_X1": (
                    df_simulation.select((pl.col("freq_rook_X1") - 5) ** 2).sum()
                    / simulations
                ).item(),
                "freq_rook_X2": (
                    df_simulation.select((pl.col("freq_rook_X2") - 6) ** 2).sum()
                    / simulations
                ).item(),
                "freq_rook_X3": (
                    df_simulation.select((pl.col("freq_rook_X3") - 7) ** 2).sum()
                    / simulations
                ).item(),
                "freq_rook_rho": (
                    df_simulation.select((pl.col("freq_rook_rho") - rho) ** 2).sum()
                    / simulations
                ).item(),
                # Frequency Queen
                "freq_queen_intercept": (
                    df_simulation.select(
                        (pl.col("freq_queen_intercept") - 4) ** 2
                    ).sum()
                    / simulations
                ).item(),
                "freq_queen_X1": (
                    df_simulation.select((pl.col("freq_queen_X1") - 5) ** 2).sum()
                    / simulations
                ).item(),
                "freq_queen_X2": (
                    df_simulation.select((pl.col("freq_queen_X2") - 6) ** 2).sum()
                    / simulations
                ).item(),
                "freq_queen_X3": (
                    df_simulation.select((pl.col("freq_queen_X3") - 7) ** 2).sum()
                    / simulations
                ).item(),
                "freq_queen_rho": (
                    df_simulation.select((pl.col("freq_queen_rho") - rho) ** 2).sum()
                    / simulations
                ).item(),
                # Frequency KNN6
                "freq_knn6_intercept": (
                    df_simulation.select((pl.col("freq_knn6_intercept") - 1) ** 2).sum()
                    / simulations
                ).item(),
                "freq_knn6_X1": (
                    df_simulation.select((pl.col("freq_knn6_X1") - 5) ** 2).sum()
                    / simulations
                ).item(),
                "freq_knn6_X2": (
                    df_simulation.select((pl.col("freq_knn6_X2") - 6) ** 2).sum()
                    / simulations
                ).item(),
                "freq_knn6_X3": (
                    df_simulation.select((pl.col("freq_knn6_X3") - 7) ** 2).sum()
                    / simulations
                ).item(),
                "freq_knn6_rho": (
                    df_simulation.select((pl.col("freq_knn6_rho") - rho) ** 2).sum()
                    / simulations
                ).item(),
                # Frequency Base (no spatial weight)
                "freq_base_intercept": (
                    df_simulation.select((pl.col("freq_base_intercept") - 4) ** 2).sum()
                    / simulations
                ).item(),
                "freq_base_X1": (
                    df_simulation.select((pl.col("freq_base_X1") - 5) ** 2).sum()
                    / simulations
                ).item(),
                "freq_base_X2": (
                    df_simulation.select((pl.col("freq_base_X2") - 6) ** 2).sum()
                    / simulations
                ).item(),
                "freq_base_X3": (
                    df_simulation.select((pl.col("freq_base_X3") - 7) ** 2).sum()
                    / simulations
                ).item(),
                # Tensor regression
                "freq_tensor_intercept": (
                    df_simulation.select(
                        (pl.col("freq_tensor_intercept") - 4) ** 2
                    ).sum()
                    / simulations
                ).item(),
                "freq_tensor_X1": (
                    df_simulation.select((pl.col("freq_tensor_X1") - 5) ** 2).sum()
                    / simulations
                ).item(),
                "freq_tensor_X2": (
                    df_simulation.select((pl.col("freq_tensor_X2") - 6) ** 2).sum()
                    / simulations
                ).item(),
                "freq_tensor_X3": (
                    df_simulation.select((pl.col("freq_tensor_X3") - 7) ** 2).sum()
                    / simulations
                ).item(),
            }

        return df_simulation

    def bayes_reg(self, data: pd.DataFrame, weight: str):
        y_true = data["y_true"].values
        X_1 = data["X_1"].values
        X_2 = data["X_2"].values
        X_3 = data["X_3"].values
        w = data[weight].values

        with pm.Model() as model:
            # Define Priors
            sigma = pm.HalfCauchy("sigma", beta=10)
            intercept = pm.Normal("intercept", 0, sigma=20)
            beta_1 = pm.Normal("X_1", 0, sigma=10)
            beta_2 = pm.Normal("X_2", 0, sigma=10)
            beta_3 = pm.Normal("X_3", 0, sigma=10)
            rho = pm.Normal("rho", 0, sigma=10)

            # Define likelihood
            likelihood = pm.Normal(
                "y_true",
                mu=intercept + beta_1 * X_1 + beta_2 * X_2 + beta_3 * X_3 + rho * w,
                sigma=sigma,
                observed=y_true,
            )

            idata = pm.sample(chains=10, progressbar=False)

        return idata

    def freq_reg(self, data: pd.DataFrame, weights: str):
        if weights == "":
            var = ["X_1", "X_2", "X_3"]
            xb = data[var].values.reshape(-1, 3)
        else:
            var = ["X_1", "X_2", "X_3", weights]
            xb = data[var].values.reshape(-1, 4)
        y_true = data["y_true"].values.reshape(-1, 1)
        X = sm.add_constant(xb)
        return sm.OLS(y_true, X).fit()

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
