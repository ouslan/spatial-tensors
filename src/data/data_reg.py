import logging
import os

import arviz as az
import bambi as bmb
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import requests
from pysal.lib import weights
from shapely import wkt
from shapely.geometry import Polygon
from spreg import Panel_FE_Lag, dgp_lag
from tqdm import tqdm

from ..sql.models import get_conn, init_dp03_table


class SpatialReg:
    def __init__(
        self,
        n,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        self.shape = self.spatial_shape(n)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            filename=log_file,
        )
        self.saving_dir = saving_dir
        self.data_file = database_file
        self.conn = get_conn(self.data_file)

    def spatial_data(self, rho: float, t: int):
        # Number of observations
        gdf = self.spatial_df()
        n_obs = len(gdf)

        # Construct X matrix (with a constant term)
        X = np.ones((n_obs, 4))
        X[:, 1] = np.random.normal(size=n_obs)  # Random variable
        X[:, 2] = np.random.normal(size=n_obs)
        X[:, 3] = np.random.normal(size=n_obs)

        # Define beta coefficients
        beta = np.array([4, 5, 6, 7])

        # Compute xb
        xb = X @ beta
        xb = xb.reshape(-1, 1)
        u = np.random.normal(size=n_obs).reshape(-1, 1)

        wr = weights.contiguity.Rook.from_dataframe(gdf, use_index=False)
        wq = weights.contiguity.Queen.from_dataframe(gdf, use_index=False)
        wk6 = weights.KNN.from_dataframe(gdf, k=6, use_index=False)
        wr = wr.transform = "r"
        wq.transform = "r"
        wk6.transform = "r"

        y_d = dgp_lag(u, xb, wq, rho=rho)

        gdf["y_d"] = y_d

        gdf["X_1"] = X[:, 1]
        gdf["X_2"] = X[:, 2]
        gdf["X_3"] = X[:, 3]
        gdf["w_rook"] = weights.lag_spatial(wr, y_d)
        gdf["w_queen"] = weights.lag_spatial(wq, y_d)
        gdf["w_knn6"] = weights.lag_spatial(wk6, y_d)
        gdf["time"] = t
        return gdf

    def spatial_panel(self, time, rho):
        gdf = gpd.GeoDataFrame(
            columns=[
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
        for t in range(0, time):
            # Remove columns with all NA values from gdf and tmp
            tmp = self.spatial_data(rho=rho, t=t)
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

    def spatial_shape(self, n):
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
        return gdf

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=zip%20code%20tabulation%20area:*"
        df = pl.DataFrame(requests.get(url).json())

        # get names from DataFrame
        names = df.select(pl.col("column_0")).transpose()
        names = names.to_dicts().pop()
        names = dict((k, v.lower()) for k, v in names.items())

        # Pivot table
        df = df.drop("column_0").transpose()
        return df.rename(names).with_columns(year=pl.lit(year))

    def pull_dp03(self) -> pl.DataFrame:
        if "DP03Table" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_dp03_table(self.data_file)
        for _year in range(2011, 2024):
            if (
                self.conn.sql(f"SELECT * FROM 'DP03Table' WHERE year={_year}")
                .df()
                .empty
            ):
                logging.info(f"pulling {_year} data")
                tmp = self.pull_query(
                    params=[
                        "DP03_0001E",
                        "DP03_0008E",
                        "DP03_0009E",
                        "DP03_0014E",
                        "DP03_0016E",
                        "DP03_0019E",
                        "DP03_0025E",
                        "DP03_0051E",
                        "DP03_0052E",
                        "DP03_0053E",
                        "DP03_0054E",
                        "DP03_0055E",
                        "DP03_0056E",
                        "DP03_0057E",
                        "DP03_0058E",
                        "DP03_0059E",
                        "DP03_0060E",
                        "DP03_0061E",
                        "DP03_0070E",
                        "DP03_0074E",
                    ],
                    year=_year,
                )
                tmp = tmp.rename(
                    {
                        "dp03_0001e": "total_population",
                        "dp03_0008e": "in_labor_force",
                        "dp03_0009e": "unemployment",
                        "dp03_0014e": "own_children6",
                        "dp03_0016e": "own_children17",
                        "dp03_0019e": "commute_car",
                        "dp03_0025e": "commute_time",
                        "dp03_0051e": "total_house",
                        "dp03_0052e": "inc_less_10k",
                        "dp03_0053e": "inc_10k_15k",
                        "dp03_0054e": "inc_15k_25k",
                        "dp03_0055e": "inc_25k_35k",
                        "dp03_0056e": "inc_35k_50k",
                        "dp03_0057e": "inc_50k_75k",
                        "dp03_0058e": "inc_75k_100k",
                        "dp03_0059e": "inc_100k_150k",
                        "dp03_0060e": "inc_150k_200k",
                        "dp03_0061e": "inc_more_200k",
                        "dp03_0070e": "with_social_security",
                        "dp03_0074e": "food_stamp",
                    }
                )
                tmp = tmp.rename({"zip code tabulation area": "zipcode"})
                self.conn.sql("INSERT INTO 'DP03Table' BY NAME SELECT * FROM tmp")
                logging.info(f"succesfully inserting {_year}")
                # except:
                #     logging.warning(f"The ACS for {_year} is not availabe")
                #     continue
            else:
                logging.info(f"data for {_year} is in the database")
                continue
        return self.conn.sql("SELECT * FROM 'DP03Table';").pl()

    def make_spatial_table(self) -> pd.DataFrame:
        # initiiate the database tables
        if "zipstable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            # Download the shape files
            if not os.path.exists(f"{self.saving_dir}external/zips_shape.zip"):
                self.pull_file(
                    url="https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip",
                    filename=f"{self.saving_dir}external/zips_shape.zip",
                )
                logging.info("Downloaded zipcode shape files")

            # Process and insert the shape files
            gdf = gpd.read_file(f"{self.saving_dir}external/zips_shape.zip")
            gdf = gdf[gdf["ZCTA5CE20"].str.startswith("00")]
            gdf = gdf.rename(columns={"ZCTA5CE20": "zipcode"}).reset_index()
            gdf = gdf[["zipcode", "geometry"]]
            gdf["zipcode"] = gdf["zipcode"].str.strip()
            df = gdf.drop(columns="geometry")
            geometry = gdf["geometry"].apply(lambda geom: geom.wkt)
            df["geometry"] = geometry
            self.conn.execute("CREATE TABLE zipstable AS SELECT * FROM df")
            logging.info(
                f"The zipstable is empty inserting {self.saving_dir}external/cousub.zip"
            )
        return self.conn.sql("SELECT * FROM zipstable;").df()

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

    def pull_file(self, url: str, filename: str, verify: bool = True) -> None:
        """
        Pulls a file from a URL and saves it in the filename. Used by the class to pull external files.

        Parameters
        ----------
        url: str
            The URL to pull the file from.
        filename: str
            The filename to save the file to.
        verify: bool
            If True, verifies the SSL certificate. If False, does not verify the SSL certificate.

        Returns
        -------
        None
        """
        chunk_size = 10 * 1024 * 1024

        with requests.get(url, stream=True, verify=verify) as response:
            total_size = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as bar:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            bar.update(
                                len(chunk)
                            )  # Update the progress bar with the size of the chunks
