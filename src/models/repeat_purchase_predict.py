from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.io import read_table, write_csv


NUMERIC_FEATURES = [
    "first_order_value",
    "first_item_count",
    "first_freight_value",
    "first_avg_installments",
    "first_late_delivery",
    "first_delivery_days",
    "first_order_month",
]

CATEGORICAL_FEATURES = [
    "first_customer_state",
    "first_product_category",
    "first_payment_type",
]

TARGET = "repeat_purchase_label"


def clean_feature_name(feature_name: str) -> str:
    cleaned = feature_name.replace("numeric__", "").replace("categorical__", "")
    cleaned = cleaned.replace("first_product_category_", "first_category=")
    cleaned = cleaned.replace("first_customer_state_", "state=")
    cleaned = cleaned.replace("first_payment_type_", "payment=")
    return cleaned


def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def run_repeat_purchase_model(processed_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    customers = read_table(processed_dir, "customer_repeat_features")
    model_df = customers.dropna(subset=[TARGET]).copy()

    if model_df[TARGET].nunique() < 2:
        raise ValueError("Repeat purchase model needs both repeat and non-repeat customers.")

    x = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_df[TARGET].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = build_model()
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else None,
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "repeat_rate": float(y.mean()),
        "test_rows": int(len(y_test)),
    }

    scored = model_df[["customer_unique_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()
    scored["repeat_probability"] = model.predict_proba(x)[:, 1]
    scored["propensity_percentile"] = scored["repeat_probability"].rank(pct=True, method="first")
    scored["propensity_band"] = np.select(
        [
            scored["propensity_percentile"] >= 0.90,
            scored["propensity_percentile"] >= 0.60,
        ],
        ["High", "Medium"],
        default="Low",
    )

    state_summary = (
        scored.groupby("first_customer_state", as_index=False)
        .agg(
            customers=("customer_unique_id", "nunique"),
            actual_repeat_rate=(TARGET, "mean"),
            avg_repeat_probability=("repeat_probability", "mean"),
        )
        .sort_values("avg_repeat_probability", ascending=False)
    )
    category_summary = (
        scored.groupby("first_product_category", as_index=False)
        .agg(
            customers=("customer_unique_id", "nunique"),
            actual_repeat_rate=(TARGET, "mean"),
            avg_repeat_probability=("repeat_probability", "mean"),
        )
        .sort_values("avg_repeat_probability", ascending=False)
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": classifier.coef_[0],
            }
        )
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
    )

    baseline_repeat_rate = float(scored[TARGET].mean())
    top_k_rows = []
    sorted_scored = scored.sort_values("repeat_probability", ascending=False)
    for top_fraction in [0.05, 0.10, 0.20]:
        selected = sorted_scored.head(max(1, int(len(sorted_scored) * top_fraction)))
        repeat_rate_at_k = float(selected[TARGET].mean())
        top_k_rows.append(
            {
                "population_slice": f"Top {int(top_fraction * 100)}%",
                "customers": int(len(selected)),
                "repeat_rate": repeat_rate_at_k,
                "lift_vs_baseline": repeat_rate_at_k / baseline_repeat_rate if baseline_repeat_rate else None,
            }
        )
    top_k_summary = pd.DataFrame(top_k_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scored.sort_values("repeat_probability", ascending=False), output_dir, "repeat_purchase_predictions.csv")
    write_csv(state_summary, output_dir, "repeat_state_summary.csv")
    write_csv(category_summary, output_dir, "repeat_category_summary.csv")
    write_csv(feature_importance, output_dir, "repeat_feature_importance.csv")
    write_csv(top_k_summary, output_dir, "repeat_topk_summary.csv")
    (output_dir / "repeat_purchase_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return scored, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a repeat purchase propensity model.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, metrics = run_repeat_purchase_model(args.processed_dir, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
