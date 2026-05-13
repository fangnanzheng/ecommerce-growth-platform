from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
from pathlib import Path
from typing import Dict

from src.data_pipeline.sql_queries import (
    CATEGORY_DAILY_SALES_SQL,
    CUSTOMER_REPEAT_FEATURES_SQL,
    CUSTOMER_FEATURES_SQL,
    FACT_ORDERS_SQL,
    STATE_CATEGORY_DAILY_SALES_SQL,
)


RAW_TABLES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def raw_files_available(raw_dir: Path) -> bool:
    return all((raw_dir / filename).exists() for filename in RAW_TABLES.values())


def spark_runtime_available() -> bool:
    if importlib.util.find_spec("pyspark") is None or shutil.which("java") is None:
        return False

    if platform.system().lower() != "windows":
        return True

    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not hadoop_home:
        return False
    return (Path(hadoop_home) / "bin" / "winutils.exe").exists()


def create_spark_session(app_name: str = "ecommerce-growth-platform"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_raw_tables(spark, raw_dir: Path) -> Dict[str, object]:
    tables = {}
    for table_name, filename in RAW_TABLES.items():
        path = raw_dir / filename
        df = spark.read.csv(str(path), header=True, inferSchema=True, multiLine=True, escape='"')
        df.createOrReplaceTempView(table_name)
        tables[table_name] = df
    return tables


def write_parquet(df, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.coalesce(1).write.mode("overwrite").parquet(str(output_path))


def run_spark_pipeline(raw_dir: Path, processed_dir: Path) -> None:
    if not raw_files_available(raw_dir):
        missing = [filename for filename in RAW_TABLES.values() if not (raw_dir / filename).exists()]
        raise FileNotFoundError(f"Missing raw files in {raw_dir}: {missing}")

    spark = create_spark_session()
    try:
        load_raw_tables(spark, raw_dir)

        fact_orders = spark.sql(FACT_ORDERS_SQL)
        fact_orders.createOrReplaceTempView("fact_orders")

        customer_features = spark.sql(CUSTOMER_FEATURES_SQL)
        category_daily_sales = spark.sql(CATEGORY_DAILY_SALES_SQL)
        state_category_daily_sales = spark.sql(STATE_CATEGORY_DAILY_SALES_SQL)
        customer_repeat_features = spark.sql(CUSTOMER_REPEAT_FEATURES_SQL)

        write_parquet(fact_orders, processed_dir / "fact_orders")
        write_parquet(customer_features, processed_dir / "customer_features")
        write_parquet(category_daily_sales, processed_dir / "category_daily_sales")
        write_parquet(state_category_daily_sales, processed_dir / "state_category_daily_sales")
        write_parquet(customer_repeat_features, processed_dir / "customer_repeat_features")
    finally:
        spark.stop()


def run_pandas_pipeline(raw_dir: Path, processed_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    if not raw_files_available(raw_dir):
        missing = [filename for filename in RAW_TABLES.values() if not (raw_dir / filename).exists()]
        raise FileNotFoundError(f"Missing raw files in {raw_dir}: {missing}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(raw_dir / RAW_TABLES["customers"])
    orders = pd.read_csv(raw_dir / RAW_TABLES["orders"])
    order_items = pd.read_csv(raw_dir / RAW_TABLES["order_items"])
    order_payments = pd.read_csv(raw_dir / RAW_TABLES["order_payments"])
    products = pd.read_csv(raw_dir / RAW_TABLES["products"])
    category_translation = pd.read_csv(raw_dir / RAW_TABLES["category_translation"])

    payment_summary = (
        order_payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            avg_installments=("payment_installments", "mean"),
            primary_payment_type=("payment_type", "first"),
        )
    )

    product_lookup = products.merge(category_translation, on="product_category_name", how="left")
    product_lookup["product_category"] = product_lookup["product_category_name_english"].fillna(
        product_lookup["product_category_name"]
    )

    items_enriched = order_items.merge(
        product_lookup[["product_id", "product_category"]], on="product_id", how="left"
    )
    item_summary = (
        items_enriched.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            distinct_product_count=("product_id", "nunique"),
            seller_count=("seller_id", "nunique"),
            item_revenue=("price", "sum"),
            freight_value=("freight_value", "sum"),
            product_category=("product_category", "first"),
        )
    )

    fact_orders = (
        orders.merge(customers, on="customer_id", how="inner")
        .merge(item_summary, on="order_id", how="left")
        .merge(payment_summary, on="order_id", how="left")
    )

    timestamp_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in timestamp_columns:
        fact_orders[column] = pd.to_datetime(fact_orders[column], errors="coerce")

    fact_orders["item_count"] = fact_orders["item_count"].fillna(0).astype(int)
    fact_orders["distinct_product_count"] = fact_orders["distinct_product_count"].fillna(0).astype(int)
    fact_orders["seller_count"] = fact_orders["seller_count"].fillna(0).astype(int)
    fact_orders["item_revenue"] = fact_orders["item_revenue"].fillna(0.0)
    fact_orders["freight_value"] = fact_orders["freight_value"].fillna(0.0)
    fact_orders["payment_value"] = fact_orders["payment_value"].fillna(
        fact_orders["item_revenue"] + fact_orders["freight_value"]
    )
    fact_orders["avg_installments"] = fact_orders["avg_installments"].fillna(1.0)
    fact_orders["primary_payment_type"] = fact_orders["primary_payment_type"].fillna("unknown")
    fact_orders["product_category"] = fact_orders["product_category"].fillna("unknown")
    fact_orders["is_late_delivery"] = (
        fact_orders["order_delivered_customer_date"] > fact_orders["order_estimated_delivery_date"]
    ).fillna(False).astype(int)

    fact_orders = fact_orders.rename(
        columns={
            "order_purchase_timestamp": "order_purchase_ts",
            "order_delivered_customer_date": "delivered_customer_ts",
            "order_estimated_delivery_date": "estimated_delivery_ts",
        }
    )

    fact_orders = fact_orders[
        [
            "order_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
            "order_status",
            "order_purchase_ts",
            "delivered_customer_ts",
            "estimated_delivery_ts",
            "item_count",
            "distinct_product_count",
            "seller_count",
            "item_revenue",
            "freight_value",
            "payment_value",
            "avg_installments",
            "primary_payment_type",
            "product_category",
            "is_late_delivery",
        ]
    ].dropna(subset=["order_purchase_ts"])

    valid_statuses = {"delivered", "shipped", "invoiced", "processing", "approved"}
    valid_orders = fact_orders[fact_orders["order_status"].isin(valid_statuses)].copy()
    snapshot_date = valid_orders["order_purchase_ts"].max()

    customer_features = (
        valid_orders.groupby("customer_unique_id")
        .agg(
            order_count=("order_id", "nunique"),
            total_revenue=("payment_value", "sum"),
            avg_order_value=("payment_value", "mean"),
            total_items=("item_count", "sum"),
            late_delivery_rate=("is_late_delivery", "mean"),
            first_purchase_ts=("order_purchase_ts", "min"),
            last_purchase_ts=("order_purchase_ts", "max"),
            category_count=("product_category", "nunique"),
        )
        .reset_index()
    )
    customer_features["recency_days"] = (
        snapshot_date - customer_features["last_purchase_ts"]
    ).dt.days
    customer_features["tenure_days"] = (
        customer_features["last_purchase_ts"] - customer_features["first_purchase_ts"]
    ).dt.days
    customer_features["churn_label"] = (customer_features["recency_days"] > 120).astype(int)

    category_daily_sales = (
        valid_orders.assign(order_date=valid_orders["order_purchase_ts"].dt.date)
        .groupby(["order_date", "product_category"], as_index=False)
        .agg(orders=("order_id", "nunique"), revenue=("payment_value", "sum"))
    )

    state_category_daily_sales = (
        valid_orders.assign(order_date=valid_orders["order_purchase_ts"].dt.date)
        .groupby(["order_date", "customer_state", "product_category"], as_index=False)
        .agg(orders=("order_id", "nunique"), revenue=("payment_value", "sum"))
    )

    first_orders = (
        valid_orders.sort_values(["customer_unique_id", "order_purchase_ts"])
        .groupby("customer_unique_id", as_index=False)
        .first()
    )
    lifetime_orders = (
        valid_orders.groupby("customer_unique_id", as_index=False)
        .agg(lifetime_orders=("order_id", "nunique"))
    )
    customer_repeat_features = first_orders.merge(lifetime_orders, on="customer_unique_id", how="left")
    customer_repeat_features["first_delivery_days"] = (
        customer_repeat_features["delivered_customer_ts"] - customer_repeat_features["order_purchase_ts"]
    ).dt.days
    customer_repeat_features["first_order_month"] = customer_repeat_features["order_purchase_ts"].dt.month
    customer_repeat_features["repeat_purchase_label"] = (
        customer_repeat_features["lifetime_orders"] > 1
    ).astype(int)
    customer_repeat_features = customer_repeat_features.rename(
        columns={
            "customer_state": "first_customer_state",
            "product_category": "first_product_category",
            "primary_payment_type": "first_payment_type",
            "payment_value": "first_order_value",
            "item_count": "first_item_count",
            "freight_value": "first_freight_value",
            "avg_installments": "first_avg_installments",
            "is_late_delivery": "first_late_delivery",
        }
    )[
        [
            "customer_unique_id",
            "first_customer_state",
            "first_product_category",
            "first_payment_type",
            "first_order_value",
            "first_item_count",
            "first_freight_value",
            "first_avg_installments",
            "first_late_delivery",
            "first_delivery_days",
            "first_order_month",
            "repeat_purchase_label",
        ]
    ]
    customer_repeat_features["first_delivery_days"] = customer_repeat_features[
        "first_delivery_days"
    ].fillna(customer_repeat_features["first_delivery_days"].median())

    experiment_base = valid_orders.dropna(subset=["payment_value"]).copy()
    experiment_base = experiment_base.sort_values("order_purchase_ts").tail(min(len(experiment_base), 20000))
    experiment_base["variant"] = np.where(
        pd.util.hash_pandas_object(experiment_base["order_id"], index=False) % 2 == 0,
        "control",
        "treatment",
    )
    median_value = experiment_base["payment_value"].median()
    experiment_base["converted"] = (experiment_base["payment_value"] >= median_value).astype(int)
    ab_test_sample = experiment_base[["variant", "converted", "payment_value"]].rename(
        columns={"payment_value": "order_value"}
    )

    fact_orders.to_csv(processed_dir / "fact_orders.csv", index=False)
    customer_features.to_csv(processed_dir / "customer_features.csv", index=False)
    category_daily_sales.to_csv(processed_dir / "category_daily_sales.csv", index=False)
    state_category_daily_sales.to_csv(processed_dir / "state_category_daily_sales.csv", index=False)
    customer_repeat_features.to_csv(processed_dir / "customer_repeat_features.csv", index=False)
    ab_test_sample.to_csv(processed_dir / "ab_test_sample.csv", index=False)


def generate_sample_data(processed_dir: Path, output_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = np.array(["health_beauty", "sports_leisure", "computers", "furniture", "watches"])
    states = np.array(["SP", "RJ", "MG", "RS", "PR"])
    dates = pd.date_range("2018-01-01", periods=180, freq="D")

    customers = [f"cust_{i:04d}" for i in range(1, 501)]
    orders = []
    for order_id in range(1, 1201):
        purchase_ts = rng.choice(dates) + pd.to_timedelta(int(rng.integers(0, 24)), unit="h")
        category = str(rng.choice(categories))
        revenue = round(float(rng.gamma(shape=3.0, scale=45.0)), 2)
        freight = round(float(rng.uniform(8, 45)), 2)
        delivered_days = int(rng.integers(2, 18))
        estimated_days = int(rng.integers(5, 14))
        orders.append(
            {
                "order_id": f"order_{order_id:05d}",
                "customer_unique_id": str(rng.choice(customers)),
                "customer_city": "sao paulo",
                "customer_state": str(rng.choice(states)),
                "order_status": "delivered",
                "order_purchase_ts": purchase_ts,
                "delivered_customer_ts": purchase_ts + pd.to_timedelta(delivered_days, unit="D"),
                "estimated_delivery_ts": purchase_ts + pd.to_timedelta(estimated_days, unit="D"),
                "item_count": int(rng.integers(1, 5)),
                "distinct_product_count": int(rng.integers(1, 4)),
                "seller_count": int(rng.integers(1, 3)),
                "item_revenue": revenue,
                "freight_value": freight,
                "payment_value": revenue + freight,
                "avg_installments": float(rng.integers(1, 7)),
                "primary_payment_type": str(rng.choice(["credit_card", "boleto", "voucher"])),
                "product_category": category,
                "is_late_delivery": int(delivered_days > estimated_days),
            }
        )

    fact_orders = pd.DataFrame(orders)
    snapshot_date = fact_orders["order_purchase_ts"].max()

    customer_features = (
        fact_orders.groupby("customer_unique_id")
        .agg(
            order_count=("order_id", "nunique"),
            total_revenue=("payment_value", "sum"),
            avg_order_value=("payment_value", "mean"),
            total_items=("item_count", "sum"),
            late_delivery_rate=("is_late_delivery", "mean"),
            first_purchase_ts=("order_purchase_ts", "min"),
            last_purchase_ts=("order_purchase_ts", "max"),
            category_count=("product_category", "nunique"),
        )
        .reset_index()
    )
    customer_features["recency_days"] = (snapshot_date - customer_features["last_purchase_ts"]).dt.days
    customer_features["tenure_days"] = (
        customer_features["last_purchase_ts"] - customer_features["first_purchase_ts"]
    ).dt.days
    customer_features["churn_label"] = (customer_features["recency_days"] > 90).astype(int)

    category_daily_sales = (
        fact_orders.assign(order_date=fact_orders["order_purchase_ts"].dt.date)
        .groupby(["order_date", "product_category"])
        .agg(orders=("order_id", "nunique"), revenue=("payment_value", "sum"))
        .reset_index()
    )

    state_category_daily_sales = (
        fact_orders.assign(order_date=fact_orders["order_purchase_ts"].dt.date)
        .groupby(["order_date", "customer_state", "product_category"])
        .agg(orders=("order_id", "nunique"), revenue=("payment_value", "sum"))
        .reset_index()
    )

    first_orders = (
        fact_orders.sort_values(["customer_unique_id", "order_purchase_ts"])
        .groupby("customer_unique_id", as_index=False)
        .first()
    )
    lifetime_orders = (
        fact_orders.groupby("customer_unique_id", as_index=False)
        .agg(lifetime_orders=("order_id", "nunique"))
    )
    customer_repeat_features = first_orders.merge(lifetime_orders, on="customer_unique_id", how="left")
    customer_repeat_features["first_delivery_days"] = (
        customer_repeat_features["delivered_customer_ts"] - customer_repeat_features["order_purchase_ts"]
    ).dt.days
    customer_repeat_features["first_order_month"] = customer_repeat_features["order_purchase_ts"].dt.month
    customer_repeat_features["repeat_purchase_label"] = (
        customer_repeat_features["lifetime_orders"] > 1
    ).astype(int)
    customer_repeat_features = customer_repeat_features.rename(
        columns={
            "customer_state": "first_customer_state",
            "product_category": "first_product_category",
            "primary_payment_type": "first_payment_type",
            "payment_value": "first_order_value",
            "item_count": "first_item_count",
            "freight_value": "first_freight_value",
            "avg_installments": "first_avg_installments",
            "is_late_delivery": "first_late_delivery",
        }
    )[
        [
            "customer_unique_id",
            "first_customer_state",
            "first_product_category",
            "first_payment_type",
            "first_order_value",
            "first_item_count",
            "first_freight_value",
            "first_avg_installments",
            "first_late_delivery",
            "first_delivery_days",
            "first_order_month",
            "repeat_purchase_label",
        ]
    ]

    ab_test_sample = pd.DataFrame(
        {
            "variant": np.repeat(["control", "treatment"], 1000),
            "converted": np.concatenate(
                [rng.binomial(1, 0.112, 1000), rng.binomial(1, 0.137, 1000)]
            ),
            "order_value": np.concatenate(
                [rng.gamma(3.0, 42.0, 1000), rng.gamma(3.2, 43.0, 1000)]
            ).round(2),
        }
    )

    fact_orders.to_csv(processed_dir / "fact_orders.csv", index=False)
    customer_features.to_csv(processed_dir / "customer_features.csv", index=False)
    category_daily_sales.to_csv(processed_dir / "category_daily_sales.csv", index=False)
    state_category_daily_sales.to_csv(processed_dir / "state_category_daily_sales.csv", index=False)
    customer_repeat_features.to_csv(processed_dir / "customer_repeat_features.csv", index=False)
    ab_test_sample.to_csv(processed_dir / "ab_test_sample.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets for the platform.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--sample-data", action="store_true", help="Generate a local sample dataset.")
    parser.add_argument(
        "--engine",
        choices=["auto", "spark", "pandas"],
        default="auto",
        help="ETL engine. auto uses Spark when PySpark and Java are available, otherwise pandas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_data:
        generate_sample_data(args.processed_dir, args.output_dir)
        print(f"Sample processed data written to {args.processed_dir}")
        return

    if args.engine == "spark":
        run_spark_pipeline(args.raw_dir, args.processed_dir)
    elif args.engine == "pandas":
        run_pandas_pipeline(args.raw_dir, args.processed_dir)
    elif spark_runtime_available():
        run_spark_pipeline(args.raw_dir, args.processed_dir)
    else:
        print("PySpark or Java was not found. Falling back to pandas ETL for this dataset size.")
        run_pandas_pipeline(args.raw_dir, args.processed_dir)
    print(f"Processed data written to {args.processed_dir}")


if __name__ == "__main__":
    main()
