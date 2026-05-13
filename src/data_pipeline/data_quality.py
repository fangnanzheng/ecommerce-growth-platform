from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_pipeline.etl_pyspark import RAW_TABLES
from src.utils.io import read_table, write_csv


RAW_KEY_COLUMNS = {
    "customers": "customer_id",
    "orders": "order_id",
    "order_items": None,
    "order_payments": None,
    "products": "product_id",
    "category_translation": "product_category_name",
}


def status_from_check(value: bool) -> str:
    return "pass" if value else "fail"


def add_check(checks: list[dict[str, Any]], name: str, status: str, value: Any, detail: str) -> None:
    checks.append({"check": name, "status": status, "value": value, "detail": detail})


def load_raw_table(raw_dir: Path, table_name: str) -> pd.DataFrame | None:
    path = raw_dir / RAW_TABLES[table_name]
    if not path.exists():
        return None
    return pd.read_csv(path)


def run_data_quality_checks(raw_dir: Path, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    raw_tables: dict[str, pd.DataFrame] = {}

    for table_name, filename in RAW_TABLES.items():
        path = raw_dir / filename
        exists = path.exists()
        add_check(
            checks,
            f"raw_file_exists__{filename}",
            status_from_check(exists),
            exists,
            f"Required raw file at {path}",
        )
        if exists:
            df = pd.read_csv(path)
            raw_tables[table_name] = df
            add_check(checks, f"raw_row_count__{table_name}", "pass", int(len(df)), "Raw table row count")

            key_column = RAW_KEY_COLUMNS.get(table_name)
            if key_column:
                duplicate_count = int(df[key_column].duplicated().sum())
                add_check(
                    checks,
                    f"raw_duplicate_key_count__{table_name}",
                    status_from_check(duplicate_count == 0),
                    duplicate_count,
                    f"Duplicate count for {key_column}",
                )

            null_rate = df.isna().mean().sort_values(ascending=False).head(5)
            for column, rate in null_rate.items():
                add_check(
                    checks,
                    f"raw_null_rate__{table_name}__{column}",
                    "warn" if rate > 0.25 else "pass",
                    round(float(rate), 4),
                    "Top raw missing-value rates",
                )

    if {"customers", "orders"}.issubset(raw_tables):
        missing_customer = ~raw_tables["orders"]["customer_id"].isin(raw_tables["customers"]["customer_id"])
        add_check(
            checks,
            "referential_integrity__orders_to_customers",
            status_from_check(int(missing_customer.sum()) == 0),
            int(missing_customer.sum()),
            "Orders whose customer_id is missing from customers",
        )

    if {"orders", "order_items"}.issubset(raw_tables):
        missing_order_items = ~raw_tables["order_items"]["order_id"].isin(raw_tables["orders"]["order_id"])
        add_check(
            checks,
            "referential_integrity__items_to_orders",
            status_from_check(int(missing_order_items.sum()) == 0),
            int(missing_order_items.sum()),
            "Order item rows whose order_id is missing from orders",
        )

    if {"orders", "order_payments"}.issubset(raw_tables):
        missing_payments = ~raw_tables["order_payments"]["order_id"].isin(raw_tables["orders"]["order_id"])
        add_check(
            checks,
            "referential_integrity__payments_to_orders",
            status_from_check(int(missing_payments.sum()) == 0),
            int(missing_payments.sum()),
            "Payment rows whose order_id is missing from orders",
        )

    if {"products", "order_items"}.issubset(raw_tables):
        missing_products = ~raw_tables["order_items"]["product_id"].isin(raw_tables["products"]["product_id"])
        add_check(
            checks,
            "referential_integrity__items_to_products",
            status_from_check(int(missing_products.sum()) == 0),
            int(missing_products.sum()),
            "Order item rows whose product_id is missing from products",
        )

    processed_tables = [
        "fact_orders",
        "customer_features",
        "category_daily_sales",
        "state_category_daily_sales",
        "customer_repeat_features",
        "ab_test_sample",
    ]
    processed_snapshots: dict[str, pd.DataFrame] = {}
    for table_name in processed_tables:
        try:
            df = read_table(processed_dir, table_name)
            processed_snapshots[table_name] = df
            add_check(checks, f"processed_row_count__{table_name}", "pass", int(len(df)), "Processed table row count")
        except FileNotFoundError:
            add_check(checks, f"processed_table_exists__{table_name}", "fail", False, "Processed output missing")

    if "fact_orders" in processed_snapshots:
        fact_orders = processed_snapshots["fact_orders"]
        fact_orders["order_purchase_ts"] = pd.to_datetime(fact_orders["order_purchase_ts"], errors="coerce")
        add_check(
            checks,
            "fact_orders__positive_payment_rate",
            "pass",
            round(float((fact_orders["payment_value"] > 0).mean()), 4),
            "Share of fact orders with positive payment value",
        )
        add_check(
            checks,
            "fact_orders__date_range",
            "pass",
            {
                "min_order_purchase_ts": str(fact_orders["order_purchase_ts"].min()),
                "max_order_purchase_ts": str(fact_orders["order_purchase_ts"].max()),
            },
            "Observed order purchase timestamp range",
        )

    if "customer_repeat_features" in processed_snapshots:
        repeat = processed_snapshots["customer_repeat_features"]
        repeat_rate = float(repeat["repeat_purchase_label"].mean())
        add_check(
            checks,
            "customer_repeat_features__repeat_rate",
            "pass",
            round(repeat_rate, 4),
            "Observed repeat purchase target rate",
        )

    status_counts = pd.Series([check["status"] for check in checks]).value_counts().to_dict()
    report = {
        "overall_status": "fail" if status_counts.get("fail", 0) else "pass",
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "checks": checks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(pd.DataFrame(checks), output_dir, "data_quality_summary.csv")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw and processed data quality checks.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_data_quality_checks(args.raw_dir, args.processed_dir, args.output_dir)
    print(json.dumps({"overall_status": report["overall_status"], "status_counts": report["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
