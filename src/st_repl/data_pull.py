import logging
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import polars as pl
from CensusForge import CensusAPI
from jp_tools import download


class DataPull:
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            filename=log_file,
        )
        self.saving_dir = saving_dir
        self.data_file = database_file
        self.conn = duckdb.connect()

    def make_spatial_table(self) -> pd.DataFrame:
        # initiiate the database tables
        if "zipstable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            # Download the shape files
            if not os.path.exists(f"{self.saving_dir}external/zips_shape.zip"):
                download(
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

    def pull_dp03(self) -> pl.DataFrame:

        for _year in range(2011, 2024):
            file_path = Path(f"{self.saving_dir}raw/dp03-{_year}.parquet")

            if file_path.exists():
                continue
            else:
                logging.info(f"pulling {_year} data")
                data = CensusAPI().query(
                    dataset="acs-acs5-profile",
                    year=_year,
                    params_list=[
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
                    geography="zip code tabulation area",
                )
                df = pl.DataFrame(data)
                headers = [str(col) for col in df.row(0)]

                df = df.slice(1).rename(dict(zip(df.columns, headers)))
                df = df.rename(
                    {
                        "DP03_0001E": "total_population",
                        "DP03_0008E": "in_labor_force",
                        "DP03_0009E": "unemployment",
                        "DP03_0014E": "own_children6",
                        "DP03_0016E": "own_children17",
                        "DP03_0019E": "commute_car",
                        "DP03_0025E": "commute_time",
                        "DP03_0051E": "total_house",
                        "DP03_0052E": "inc_less_10k",
                        "DP03_0053E": "inc_10k_15k",
                        "DP03_0054E": "inc_15k_25k",
                        "DP03_0055E": "inc_25k_35k",
                        "DP03_0056E": "inc_35k_50k",
                        "DP03_0057E": "inc_50k_75k",
                        "DP03_0058E": "inc_75k_100k",
                        "DP03_0059E": "inc_100k_150k",
                        "DP03_0060E": "inc_150k_200k",
                        "DP03_0061E": "inc_more_200k",
                        "DP03_0070E": "with_social_security",
                        "DP03_0074E": "food_stamp",
                        "zip code tabulation area": "zipcode",
                    }
                )
                df = df.cast(
                    {
                        "total_population": pl.Int32,
                        "in_labor_force": pl.Int32,
                        "unemployment": pl.Int32,
                        "own_children6": pl.Int32,
                        "own_children17": pl.Int32,
                        "commute_car": pl.Int32,
                        "commute_time": pl.Float32,
                        "total_house": pl.Int32,
                        "inc_less_10k": pl.Int32,
                        "inc_10k_15k": pl.Int32,
                        "inc_15k_25k": pl.Int32,
                        "inc_25k_35k": pl.Int32,
                        "inc_35k_50k": pl.Int32,
                        "inc_50k_75k": pl.Int32,
                        "inc_75k_100k": pl.Int32,
                        "inc_100k_150k": pl.Int32,
                        "inc_150k_200k": pl.Int32,
                        "inc_more_200k": pl.Int32,
                        "with_social_security": pl.Int32,
                        "food_stamp": pl.Int32,
                        "zipcode": pl.String,
                    }
                )
                df.write_parquet(file_path)
                logging.info(f"succesfully inserting {_year}")
        return self.conn.execute(
            f"SELECT * FROM '{self.saving_dir}raw/dp03-*.parquet';"
        ).pl()
