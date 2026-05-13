from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.io import read_table, write_csv


FEATURE_COLUMNS = ["recency_days", "order_count", "total_revenue", "avg_order_value", "category_count"]


def assign_segment_names(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["value_score"] = (
        ranked["total_revenue"].rank(ascending=False)
        + ranked["order_count"].rank(ascending=False)
        + ranked["recency_days"].rank(ascending=True)
    )
    ranked = ranked.sort_values("value_score")

    labels = ["Champions", "Loyal", "Promising", "At Risk"]
    segment_name_map = {
        cluster_id: labels[min(index, len(labels) - 1)]
        for index, cluster_id in enumerate(ranked["cluster"].tolist())
    }
    summary["segment_name"] = summary["cluster"].map(segment_name_map)
    return summary


def run_segmentation(processed_dir: Path, output_dir: Path, n_clusters: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = read_table(processed_dir, "customer_features")
    model_df = customers.dropna(subset=FEATURE_COLUMNS).copy()

    if len(model_df) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} customers to run K-Means; got {len(model_df)}.")

    scaler = StandardScaler()
    features = scaler.fit_transform(model_df[FEATURE_COLUMNS])

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    model_df["cluster"] = model.fit_predict(features)

    summary = (
        model_df.groupby("cluster")
        .agg(
            customers=("customer_unique_id", "nunique"),
            recency_days=("recency_days", "mean"),
            order_count=("order_count", "mean"),
            total_revenue=("total_revenue", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            churn_rate=("churn_label", "mean"),
        )
        .reset_index()
    )
    summary = assign_segment_names(summary)
    model_df = model_df.merge(summary[["cluster", "segment_name"]], on="cluster", how="left")

    write_csv(model_df, output_dir, "segmented_customers.csv")
    write_csv(summary, output_dir, "segment_summary.csv")
    return model_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RFM-style customer segmentation.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--clusters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = run_segmentation(args.processed_dir, args.output_dir, args.clusters)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
