import logging
from pathlib import Path
import hashlib
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pymc as pd_pm  # keeping standard names or aliases clear
import pymc as pm
import statsmodels.api as sm
from libpysal import weights
from patsy import dmatrix
from sklearn.linear_model import Ridge
from spreg import dgp_lag
from jp_qcew import CleanQCEW
from jp_tools import download

from .data_pull import DataPull


class SpatialReg(DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, log_file)

        spatial_df = self.spatial_df()

        # Define spatial weight matrices
        self.wr = weights.contiguity.Rook.from_dataframe(spatial_df, use_index=False)

        self.wq = weights.contiguity.Queen.from_dataframe(spatial_df, use_index=False)
        self.wq.transform = "r"

        self.wk6 = weights.KNN.from_dataframe(spatial_df, k=6, use_index=False)
        self.wk6.transform = "r"

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

    def spatial_panel(self, time: int, rho: float, seed: int) -> pd.DataFrame:
        panels = [
            self.spatial_data(mu=2, sigma=3, rho=rho, time=t, seed=seed + t)
            for t in range(time)
        ]
        return pd.concat(panels, ignore_index=True)

    def spatial_simulation(
        self, time: int, rho: float, simulations: int, start_seed: int
    ):
        logging.getLogger("pymc").setLevel(logging.WARNING)

        col_names = [
            f"{model}_{var}"
            for model in [
                "freq_rook",
                "freq_queen",
                "freq_knn6",
                "freq_base",
                "freq_tensor",
            ]
            for var in ["X_1", "X_2", "X_3", "rho", "intercept"]
            if not (model == "freq_tensor" and var == "rho")
        ] + ["simulknations_id"]

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

    def spatial_df(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(self.make_spatial_table())
        return gdf.to_crs("EPSG:3395").assign(
            zipcode=lambda x: x["zipcode"].astype(str)
        )

    def make_spatial_table(self) -> pd.DataFrame:
        file_path = self.saving_dir / "external" / "geo-zips.parquet"
        if not file_path.exists():
            name_hash = hashlib.md5(str(file_path).encode()).hexdigest()
            temp_zip = Path(tempfile.gettempdir()) / f"census_zip_{name_hash}.zip"

            download(
                url="https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip",
                filename=temp_zip,
            )
            logging.info("Downloaded zipcode shape files")

            gdf = gpd.read_file(f"{self.saving_dir}external/zips_shape.zip")
            gdf = gdf[gdf["ZCTA5CE20"].str.startswith("00")].rename(
                columns={"ZCTA5CE20": "zipcode"}
            )
            gdf[["zipcode", "geometry"]].assign(
                zipcode=lambda x: x["zipcode"].str.strip()
            ).to_parquet(file_path)

        return gpd.read_parquet(file_path)
