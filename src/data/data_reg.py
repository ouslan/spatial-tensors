import arviz as az
import bambi as bmb
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from pysal.lib import weights
from shapely.geometry import Polygon
from spreg import dgp_lag, Panel_FE_Lag


class SpatialReg:
    def __init__(self, seed=None) -> None:
        pass

    def spatial_data(self, n, rho, t):
        l = np.arange(n)
        xs, ys = np.meshgrid(l, l)
        polys = []
        # Generate polygons
        for x, y in zip(xs.flatten(), ys.flatten()):
            poly = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
            polys.append(poly)
        # Convert to GeoSeries
        polys = gpd.GeoSeries(polys)
        gdf = gpd.GeoDataFrame(
            {
                "geometry": polys,
                "id": ["P-%s" % str(i).zfill(2) for i in range(len(polys))],
            }
        )

        # Number of observations
        n_obs = len(gdf)

        # Construct X matrix (with a constant term)
        X = np.ones((n_obs, 2))
        X[:, 1] = np.random.normal(size=n_obs)  # Random variable

        # Define beta coefficients
        beta = np.array([1, 2])

        # Compute xb
        xb = X @ beta
        xb = xb.reshape(-1, 1)
        u = np.random.normal(size=n_obs).reshape(-1, 1)

        wr = weights.contiguity.Queen.from_dataframe(gdf, use_index=False)
        wr.transform = "r"

        y_d = dgp_lag(u, xb, wr, rho=rho)

        gdf["y_d"] = y_d

        gdf["X1"] = X[:, 1]
        gdf["w_d"] = weights.lag_spatial(wr, y_d)
        gdf["time"] = t
        return gdf

    def spatial_panel(self, n, time, rho):
        gdf = gpd.GeoDataFrame(columns=["geometry", "id", "y_d", "X1", "w_d", "time"])
        for t in range(0, time):
            # Remove columns with all NA values from gdf and tmp
            tmp = self.spatial_data(rho=rho, t=t, n=n)
            # gdf = gdf.dropna(axis=1, how='all')
            # tmp = tmp.dropna(axis=1, how='all')

            gdf = pd.concat([gdf, tmp]).reset_index(drop=True)

        return gdf

    def spatial_simulation(self, n, time, rho, simulations):
        df = pl.DataFrame(
            [
                pl.Series("bayes_x", [], dtype=pl.Float64),
                pl.Series("bayes_w", [], dtype=pl.Float64),
                pl.Series("freq_x", [], dtype=pl.Float64),
                pl.Series("freq_w", [], dtype=pl.Float64),
            ]
        )
        for _ in range(0, simulations):
            gdf = self.spatial_panel(n=n, time=time, rho=rho)
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
                ]
            )
            df = pl.concat([df, df_sim], how="vertical")

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
