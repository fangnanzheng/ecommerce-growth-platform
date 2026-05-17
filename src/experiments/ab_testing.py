from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
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
    "repeat_probability",
    "target_category_affinity",
    "core_market",
]

CATEGORICAL_FEATURES = [
    "first_customer_state",
    "first_product_category",
    "first_payment_type",
]

CORE_STATES = {"SP", "RJ", "MG"}
TARGET_CATEGORY = "health_beauty"
RANDOM_SEED = 20260514
TREATMENT_PROBABILITY = 0.5
TARGET_SEGMENT_FRACTION = 0.05


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def try_build_xgb_regressor(random_state: int):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=2,
            verbosity=0,
        ), "XGBoost"
    except ImportError:
        return (
            HistGradientBoostingRegressor(
                max_iter=140,
                max_leaf_nodes=15,
                learning_rate=0.05,
                random_state=random_state,
            ),
            "sklearn HistGradientBoosting",
        )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_regression_pipeline(random_state: int) -> tuple[Pipeline, str]:
    learner, learner_name = try_build_xgb_regressor(random_state)
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", learner)]), learner_name


def load_experiment_frame(processed_dir: Path, output_dir: Path) -> pd.DataFrame:
    customers = read_table(processed_dir, "customer_repeat_features")
    predictions_path = output_dir / "repeat_purchase_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, usecols=["customer_unique_id", "repeat_probability"])
        customers = customers.merge(predictions, on="customer_unique_id", how="left")
    else:
        customers["repeat_probability"] = customers["repeat_purchase_label"].fillna(0).astype(float)

    customers["repeat_probability"] = customers["repeat_probability"].fillna(customers["repeat_probability"].median())
    customers = customers.dropna(subset=["customer_unique_id"]).copy()
    return customers


def generate_sparse_promo_dgp(
    df: pd.DataFrame,
    target_category: str = TARGET_CATEGORY,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    data = df.copy()

    repeat_score = data["repeat_probability"].fillna(data["repeat_probability"].median()).clip(0, 1)
    repeat_z = (repeat_score - repeat_score.mean()) / max(repeat_score.std(ddof=0), 1e-6)
    order_value = data["first_order_value"].fillna(data["first_order_value"].median()).clip(lower=0)
    freight_value = data["first_freight_value"].fillna(data["first_freight_value"].median()).clip(lower=0)
    delivery_days = data["first_delivery_days"].fillna(data["first_delivery_days"].median()).clip(lower=0)
    item_count = data["first_item_count"].fillna(1).clip(lower=1)
    is_core_state = data["first_customer_state"].isin(CORE_STATES).astype(int)
    is_target_category = (data["first_product_category"] == target_category).astype(int)
    is_credit_card = (data["first_payment_type"] == "credit_card").astype(int)
    is_late = data["first_late_delivery"].fillna(0).astype(int)
    data["target_category_affinity"] = is_target_category
    data["core_market"] = is_core_state

    baseline_purchase_logit = (
        -1.85
        + 0.45 * repeat_z
        + 0.45 * is_target_category
        + 0.22 * is_core_state
        + 0.12 * is_credit_card
        - 0.18 * is_late
    )
    purchase_probability = sigmoid(baseline_purchase_logit.to_numpy())
    baseline_buyer = rng.binomial(1, purchase_probability)
    expected_amount = 38 + 0.10 * order_value + 4.5 * np.log1p(item_count) + 0.04 * freight_value
    spend_noise = rng.lognormal(mean=0.0, sigma=0.95, size=len(data))
    y0 = baseline_buyer * expected_amount.to_numpy() * spend_noise

    high_repeat_threshold = repeat_score.quantile(0.85)
    responders = (
        (repeat_score >= high_repeat_threshold)
        & (data["first_customer_state"].isin(CORE_STATES))
        & (data["first_product_category"] == target_category)
    )
    tau = np.where(responders, 720 + 0.20 * np.minimum(order_value.to_numpy(), 500), 0.0)
    treatment = rng.binomial(1, TREATMENT_PROBABILITY, size=len(data))
    outcome_noise = rng.normal(0, 280, size=len(data))
    outcome_noise[treatment == 1] -= outcome_noise[treatment == 1].mean()
    outcome_noise[treatment == 0] -= outcome_noise[treatment == 0].mean()
    outcome = np.clip(y0 + treatment * tau + outcome_noise, 0, None)

    data["treatment"] = treatment
    data["synthetic_outcome"] = outcome
    data["baseline_outcome"] = y0
    data["true_tau"] = tau
    data["true_responder"] = responders.astype(int)
    data["target_category"] = target_category
    data["promo_window_days"] = 60
    data["treatment_probability"] = TREATMENT_PROBABILITY
    return data


def welch_difference_in_means(df: pd.DataFrame) -> dict:
    treated = df.loc[df["treatment"] == 1, "synthetic_outcome"].to_numpy()
    control = df.loc[df["treatment"] == 0, "synthetic_outcome"].to_numpy()
    t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)
    return {
        "method": "Difference in means",
        "estimand": "Global ATE",
        "estimate": float(treated.mean() - control.mean()),
        "standard_error": float(
            np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        ),
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant_05": bool(p_value < 0.05),
        "treated_rows": int(len(treated)),
        "control_rows": int(len(control)),
    }


def one_sample_mean_test(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    statistic = mean / se if se and not np.isnan(se) else float("nan")
    p_value = float(2 * stats.t.sf(abs(statistic), df=len(values) - 1)) if len(values) > 1 else float("nan")
    return {
        "estimate": mean,
        "standard_error": se,
        "statistic": float(statistic),
        "p_value": p_value,
        "is_significant_05": bool(p_value < 0.05),
    }


def run_r_learner_aipw(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
    target_segment_fraction: float = TARGET_SEGMENT_FRACTION,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
    model_df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["synthetic_outcome", "treatment"]).copy()
    train_idx, test_idx = train_test_split(
        model_df.index,
        test_size=0.5,
        random_state=random_seed,
        stratify=model_df["treatment"],
    )

    train = model_df.loc[train_idx].copy()
    test = model_df.loc[test_idx].copy()
    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train["synthetic_outcome"].to_numpy()
    y_test = test["synthetic_outcome"].to_numpy()
    w_train = train["treatment"].to_numpy()
    w_test = test["treatment"].to_numpy()
    e = TREATMENT_PROBABILITY

    outcome_model, learner_name = build_regression_pipeline(random_seed)
    outcome_model.fit(x_train, y_train)
    m_train = outcome_model.predict(x_train)
    m_test = outcome_model.predict(x_test)

    pseudo_tau = (y_train - m_train) / (w_train - e)
    r_weights = (w_train - e) ** 2
    tau_model, _ = build_regression_pipeline(random_seed + 1)
    tau_model.fit(x_train, pseudo_tau, model__sample_weight=r_weights)
    tau_hat = tau_model.predict(x_test)

    cutoff = float(np.quantile(tau_hat, 1 - target_segment_fraction))
    selected = tau_hat >= cutoff
    mu0_hat = m_test - e * tau_hat
    mu1_hat = mu0_hat + tau_hat
    aipw_scores = tau_hat + (w_test / e) * (y_test - mu1_hat) - ((1 - w_test) / (1 - e)) * (y_test - mu0_hat)
    selected_scores = aipw_scores[selected]
    aipw_test = one_sample_mean_test(selected_scores)

    test = test.assign(
        split="test",
        tau_hat=tau_hat,
        selected_segment=selected.astype(int),
        aipw_score=aipw_scores,
    )
    train = train.assign(split="train", tau_hat=np.nan, selected_segment=0, aipw_score=np.nan)
    scored = pd.concat([train, test], ignore_index=True)

    selected_test = test.loc[selected]
    profile = pd.DataFrame(
        [
            {
                "Segment": "All holdout customers",
                "Customers": int(len(test)),
                "Treatment rate": float(test["treatment"].mean()),
                "True responder rate": float(test["true_responder"].mean()),
                "True average effect": float(test["true_tau"].mean()),
                "Predicted CATE": float(test["tau_hat"].mean()),
            },
            {
                "Segment": f"Top {int(target_segment_fraction * 100)}% predicted CATE",
                "Customers": int(len(selected_test)),
                "Treatment rate": float(selected_test["treatment"].mean()),
                "True responder rate": float(selected_test["true_responder"].mean()),
                "True average effect": float(selected_test["true_tau"].mean()),
                "Predicted CATE": float(selected_test["tau_hat"].mean()),
            },
        ]
    )

    result = {
        "method": f"R-learner + holdout AIPW ({learner_name})",
        "estimand": f"ATE inside top {int(target_segment_fraction * 100)}% predicted CATE segment",
        "estimate": aipw_test["estimate"],
        "standard_error": aipw_test["standard_error"],
        "statistic": aipw_test["statistic"],
        "p_value": aipw_test["p_value"],
        "is_significant_05": aipw_test["is_significant_05"],
        "train_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "selected_holdout_rows": int(len(selected_test)),
        "selected_fraction": float(len(selected_test) / len(test)),
        "true_selected_ate": float(selected_test["true_tau"].mean()),
        "true_selected_responder_rate": float(selected_test["true_responder"].mean()),
        "cate_cutoff": cutoff,
        "base_learner": learner_name,
    }
    return result, scored, profile, learner_name


def summarize_dgp(df: pd.DataFrame, target_category: str, learner_name: str) -> dict:
    return {
        "experiment_type": "Semi-synthetic sparse promotion experiment",
        "unit": "customer",
        "target_category": target_category,
        "target_category_display": target_category.replace("_", " ").capitalize(),
        "promo_window_days": 60,
        "assignment": "50/50 randomized treatment assignment generated with a fixed seed",
        "outcome": "Synthetic post-promotion revenue over 60 days",
        "base_learner": learner_name,
        "customers": int(len(df)),
        "treated_customers": int(df["treatment"].sum()),
        "control_customers": int((1 - df["treatment"]).sum()),
        "true_responder_rate": float(df["true_responder"].mean()),
        "true_global_ate": float(df["true_tau"].mean()),
        "true_responder_ate": float(df.loc[df["true_responder"] == 1, "true_tau"].mean()),
        "response_rule": "repeat_probability >= 85th percentile AND state in SP/RJ/MG AND first category is health_beauty",
    }


def run_ab_analysis(processed_dir: Path, output_dir: Path) -> dict:
    customers = load_experiment_frame(processed_dir, output_dir)
    experiment = generate_sparse_promo_dgp(customers)
    naive = welch_difference_in_means(experiment)
    r_learner, scored, profile, learner_name = run_r_learner_aipw(experiment)
    dgp = summarize_dgp(experiment, TARGET_CATEGORY, learner_name)

    comparison = {
        "naive_rejects_05": naive["is_significant_05"],
        "r_learner_rejects_05": r_learner["is_significant_05"],
        "intended_lesson": "A global average-effect test can miss sparse uplift, while sample-split CATE targeting can identify a responsive segment on holdout data.",
    }
    result = {
        "dgp": dgp,
        "naive_difference_in_means": naive,
        "r_learner_aipw": r_learner,
        "comparison": comparison,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        scored[
            [
                "customer_unique_id",
                "split",
                "treatment",
                "synthetic_outcome",
                "true_tau",
                "true_responder",
                "tau_hat",
                "selected_segment",
                "aipw_score",
                "repeat_probability",
                "first_customer_state",
                "first_product_category",
                "first_order_value",
            ]
        ],
        output_dir,
        "semi_synthetic_experiment_sample.csv",
    )
    write_csv(profile, output_dir, "semi_synthetic_segment_profile.csv")
    (output_dir / "ab_test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semi-synthetic causal experiment validation.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ab_analysis(args.processed_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
