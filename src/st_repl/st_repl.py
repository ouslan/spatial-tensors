import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import statsmodels.api as sm
from libpysal import weights
from patsy import dmatrix
from shapely.geometry import box
from sklearn.linear_model import Ridge
from spreg import dgp_lag
from jp_qcew import CleanQCEW

from .data_pull import DataPull


class SpatialReg(DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        log_file: str = "data_process.log",
        grid_x: int = 10,
        grid_y: int = 10,
    ):
        super().__init__(saving_dir, log_file)
        self.grid_x = grid_x
        self.grid_y = grid_y

        spatial_df = self.spatial_df()

        # Define spatial weight matrices using the synthetic grid
        self.wr = weights.contiguity.Rook.from_dataframe(spatial_df, use_index=False)

        self.wq = weights.contiguity.Queen.from_dataframe(spatial_df, use_index=False)
        self.wq.transform = "r"

        self.wk6 = weights.KNN.from_dataframe(spatial_df, k=6, use_index=False)
        self.wk6.transform = "r"

    def make_spatial_table(self) -> gpd.GeoDataFrame:
        """Generates a synthetic regular polygon grid (GeoDataFrame) instead of loading a parquet file."""
        polygons = []
        ids = []

        id_counter = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                # Create a square polygon cell of size 1x1
                polygons.append(box(x, y, x + 1, y + 1))
                ids.append(f"{id_counter:04d}")
                id_counter += 1

        grid_gdf = gpd.GeoDataFrame(
            {"zipcode": ids, "geometry": polygons}, crs="EPSG:3395"
        )
        return grid_gdf

    def spatial_df(self) -> gpd.GeoDataFrame:
        gdf = self.make_spatial_table()
        return gdf.to_crs("EPSG:3395").assign(
            zipcode=lambda x: x["zipcode"].astype(str)
        )

    def spatial_data(
        self, mu: int, sigma: int, rho: float, time: int, seed: int
    ) -> gpd.GeoDataFrame:
        gdf = self.spatial_df().copy()
        n_obs = len(gdf)
        rng = np.random.default_rng(seed=seed)

        # Generate independent variables: X0 (intercept/ones) + 3 normal variables
        X = np.column_stack(
            [np.ones(n_obs), rng.normal(loc=mu, scale=sigma, size=(n_obs, 3))]
        )

        beta = np.array([4, 5, 6, 7])
        xb = (X @ beta).reshape(-1, 1)
        u = rng.normal(loc=0, scale=2, size=(n_obs, 1))

        y_true = dgp_lag(u, xb, self.wq, rho=rho)

        # Assign values efficiently
        gdf["y_true"] = y_true
        for i in range(1, 4):
            gdf[f"X_{i}"] = X[:, i]

        centroids = gdf.geometry.centroid
        gdf["lat"] = pd.to_numeric(centroids.x, errors="coerce")
        gdf["lon"] = pd.to_numeric(centroids.y, errors="coerce")
        gdf["centroid"] = centroids

        gdf["w_rook"] = weights.lag_spatial(self.wr, y_true)
        gdf["w_queen"] = weights.lag_spatial(self.wq, y_true)
        gdf["w_knn6"] = weights.lag_spatial(self.wk6, y_true)
        gdf["time"] = time

        return gdf.dropna(axis=1, how="all")

    def spatial_panel(self, mu:int, time: int, rho: float, sigma:float, seed: int) -> pd.DataFrame:
        panels = [
            self.spatial_data(mu=mu, sigma=sigma, rho=rho, time=t, seed=seed + t)
            for t in range(time)
        ]
        return pd.concat(panels, ignore_index=True)

    def spatial_simulation(
        self, time: int, rho: float, simulations: int, start_seed: int
    ):
        logging.getLogger("pymc").setLevel(logging.WARNING)

        sim_records = []

        for i in range(simulations):
            current_seed = start_seed + i + 1
            gdf = self.spatial_panel(time=time, rho=rho, seed=current_seed)
            df = gdf.drop(columns="geometry")

            # Tensor regression smoothing via Ridge
            formula = "X_1 + X_2 + X_3 + te(cr(lat, df=6), cr(lon, df=6), constraints='center')"
            design = dmatrix(formula, gdf)
            tensor_model = Ridge(alpha=1e-3).fit(design, gdf["y_true"])
            coef_dict = dict(zip(design.design_info.column_names, tensor_model.coef_))

            # Frequentist regressions
            res_rook = self.freq_reg(data=df, weights="w_rook")
            res_queen = self.freq_reg(data=df, weights="w_queen")
            res_knn6 = self.freq_reg(data=df, weights="w_knn6")
            res_base = self.freq_reg(data=df, weights="")

            sim_records.append(
                {
                    **{
                        f"freq_rook_{v}": res_rook.params[idx]
                        for idx, v in enumerate(
                            ["intercept", "X_1", "X_2", "X_3", "rho"]
                        )
                    },
                    **{
                        f"freq_queen_{v}": res_queen.params[idx]
                        for idx, v in enumerate(
                            ["intercept", "X_1", "X_2", "X_3", "rho"]
                        )
                    },
                    **{
                        f"freq_knn6_{v}": res_knn6.params[idx]
                        for idx, v in enumerate(
                            ["intercept", "X_1", "X_2", "X_3", "rho"]
                        )
                    },
                    **{
                        f"freq_base_{v}": res_base.params[idx]
                        for idx, v in enumerate(["intercept", "X_1", "X_2", "X_3"])
                    },
                    "freq_tensor_X1": coef_dict["X_1"],
                    "freq_tensor_X2": coef_dict["X_2"],
                    "freq_tensor_X3": coef_dict["X_3"],
                    "freq_tensor_intercept": tensor_model.intercept_,
                    "simulknations_id": i,
                }
            )
            logging.info(f"Completed Simulation #{i} successfully")

        df_simulation = pl.DataFrame(sim_records)

        # Compute MSE results
        true_vals = {"intercept": 4, "X_1": 5, "X_2": 6, "X_3": 7, "rho": rho}
        self.results = {}

        for prefix in [
            "freq_rook",
            "freq_queen",
            "freq_knn6",
            "freq_base",
            "freq_tensor",
        ]:
            for param, true_val in true_vals.items():
                col = f"{prefix}_{param}"
                if col in df_simulation.columns:
                    mse = df_simulation.select(
                        ((pl.col(col) - true_val) ** 2).sum() / simulations
                    ).item()
                    self.results[col] = mse

        return df_simulation

    def freq_reg(self, data: pd.DataFrame, weights: str):
        vars_to_use = (
            ["X_1", "X_2", "X_3"] if weights == "" else ["X_1", "X_2", "X_3", weights]
        )
        xb = data[vars_to_use].values
        y_true = data["y_true"].values.reshape(-1, 1)
        return sm.OLS(y_true, sm.add_constant(xb)).fit()

    def calculate_spatial_lag(self, df, w, column):
        return weights.lag_spatial(w, df[column].values.reshape(-1, 1))

    def quasi_data(self) -> gpd.GeoDataFrame:
        data_path = self.saving_dir / "processed" / "qcew" / "2024" / "data-2.parquet"
        if not data_path.exists():
            CleanQCEW().make_qcew_dataset()

        df_qcew = self.conn.execute(f"""
            SELECT year, qtr, phys_addr_5_zip, phys_addr_city, ui_addr_5_zip, 
                   mail_addr_5_zip, ein, first_month_employment, total_wages, 
                   second_month_employment, third_month_employment, naics_code
            FROM '{self.saving_dir}/processed/qcew/**/data-*.parquet';
        """).pl()

        df_qcew = (
            df_qcew.with_columns(
                [
                    pl.col(c).fill_null(strategy="zero")
                    for c in [
                        "first_month_employment",
                        "second_month_employment",
                        "third_month_employment",
                        "total_wages",
                    ]
                ]
            )
            .with_columns(
                total_employment=(
                    pl.col("first_month_employment")
                    + pl.col("second_month_employment")
                    + pl.col("third_month_employment")
                )
                / 3
            )
            .filter(
                (pl.col("total_employment") != 0)
                & (pl.col("total_wages") != 0)
                & (pl.col("phys_addr_city") != "")
                & (pl.col("naics_code") != "")
            )
            .with_columns(pl.col("phys_addr_city").str.to_lowercase())
            .group_by(["year", "qtr", "phys_addr_city"])
            .agg([pl.col("total_employment").sum(), pl.col("total_wages").sum()])
            .rename({"phys_addr_city": "name"})
            .to_pandas()
        )

        gdf = self.county_geom()
        accented, unaccented = "áéíóúüñÁÉÍÓÚ", "aeiouunAEIOU"
        gdf["name"] = (
            gdf["name"].str.lower().str.translate(str.maketrans(accented, unaccented))
        )

        return gpd.GeoDataFrame(pd.merge(gdf, df_qcew, on="name"), geometry="geometry")

    def quasi_panel(
        self, alpha: int, beta: int, sigma: int, rho: float, seed: int
    ) -> gpd.GeoDataFrame:

        gdf = (
            self.quasi_data()
            .sort_values(["year", "qtr", "name"])
            .to_crs("EPSG:3395")
            .reset_index(drop=True)
        )

        all_slices = []

        for year in range(2010, 2017):
            for qtr in range(1, 5):

                slice_df = gdf[(gdf["year"] == year) & (gdf["qtr"] == qtr)].reset_index(
                    drop=True
                )

                # Skip empty slices
                if (
                    len(slice_df) < 2
                ):  # Most spatial weight methods require at least 2 observations
                    continue

                rng_global = np.random.default_rng(seed=seed + (year * 10 + qtr))

                # Create slice-specific weight matrices
                try:
                    wr_slice = weights.contiguity.Rook.from_dataframe(
                        slice_df, use_index=False
                    )
                    wq_slice = weights.contiguity.Queen.from_dataframe(
                        slice_df, use_index=False
                    )
                    wq_slice.transform = "r"

                    # Ensure KNN k doesn't exceed available observations minus 1
                    k_val = min(6, len(slice_df) - 1)
                    wk6_slice = weights.KNN.from_dataframe(
                        slice_df, k=k_val, use_index=False
                    )
                    wk6_slice.transform = "r"
                except Exception:
                    # Skip if topology generation fails for disconnected/tiny slices
                    continue

                X = slice_df[["total_employment", "total_wages"]].values
                X = sm.add_constant(X)

                n_obs = len(slice_df)

                coef = np.array([200, alpha])
                xb = (X[:, :2] @ coef).reshape(-1, 1)

                u = rng_global.normal(loc=0, scale=sigma, size=n_obs).reshape(-1, 1)

                # Generate spatial lag using the slice's queen weights
                y_true = dgp_lag(u, xb, wq_slice, rho=rho, imethod="true_inv")

                if y_true is None:
                    continue

                slice_df["y_true"] = y_true
                slice_df["centroid"] = slice_df.geometry.centroid
                slice_df["lat"] = slice_df["centroid"].y
                slice_df["lon"] = slice_df["centroid"].x

                slice_df["w_rook"] = weights.lag_spatial(wr_slice, y_true)
                slice_df["w_queen"] = weights.lag_spatial(wq_slice, y_true)
                slice_df["w_knn6"] = weights.lag_spatial(wk6_slice, y_true)

                all_slices.append(slice_df)

        if not all_slices:
            raise ValueError(
                "No data slices were processed. Check your filters and dataset."
            )

        master = gpd.GeoDataFrame(pd.concat(all_slices, ignore_index=True), crs=gdf.crs)

        return master
