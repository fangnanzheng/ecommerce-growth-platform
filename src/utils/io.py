from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(data_dir: Path, table_name: str) -> pd.DataFrame:
    csv_path = data_dir / f"{table_name}.csv"
    parquet_path = data_dir / table_name

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Could not find table '{table_name}' as {csv_path} or parquet directory {parquet_path}."
    )


def write_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path
