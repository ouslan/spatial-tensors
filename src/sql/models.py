import duckdb
from pathlib import Path


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_dp03_table(db_path: str) -> None:
    current_dir = Path(__file__).parent
    sql_path = current_dir / "init_qcew_table.sql"
    with sql_path.open("r", encoding="utf-8") as file:
        dp03_script = file.read()

    conn = get_conn(db_path)
    conn.sql(dp03_script)
