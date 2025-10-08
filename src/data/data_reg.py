import logging

import arviz as az
import bambi as bmb
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import requests
from pysal.lib import weights
from shapely.geometry import Polygon
from spreg import Panel_FE_Lag, dgp_lag


class SpatialReg:
    def __init__(
        self,
        n,
        log_file: str = "data_process.log",
    ):
        self.shape = self.spatial_shape(n)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            filename=log_file,
        )

    def spatial_data(self, n, rho, t):
        # Number of observations
        gdf = self.shape
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
                pl.Series("simulations_id", [], dtype=pl.Int32),
            ]
        )
        for i in range(0, simulations):
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
