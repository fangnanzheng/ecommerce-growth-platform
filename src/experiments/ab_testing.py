from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from src.utils.io import read_table


def conversion_z_test(df: pd.DataFrame) -> dict:
    summary = df.groupby("variant")["converted"].agg(["sum", "count"]).sort_index()
    if set(summary.index) != {"control", "treatment"}:
        raise ValueError("A/B test data must contain exactly control and treatment variants.")

    counts = np.array([summary.loc["control", "sum"], summary.loc["treatment", "sum"]])
    nobs = np.array([summary.loc["control", "count"], summary.loc["treatment", "count"]])
    z_stat, p_value = proportions_ztest(count=counts, nobs=nobs, alternative="two-sided")

    control_rate = counts[0] / nobs[0]
    treatment_rate = counts[1] / nobs[1]
    return {
        "control_conversion_rate": float(control_rate),
        "treatment_conversion_rate": float(treatment_rate),
        "absolute_lift": float(treatment_rate - control_rate),
        "relative_lift": float((treatment_rate - control_rate) / control_rate) if control_rate else None,
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "is_significant_05": bool(p_value < 0.05),
    }


def order_value_t_test(df: pd.DataFrame) -> dict:
    control = df.loc[df["variant"] == "control", "order_value"].dropna()
    treatment = df.loc[df["variant"] == "treatment", "order_value"].dropna()
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    return {
        "control_mean_order_value": float(control.mean()),
        "treatment_mean_order_value": float(treatment.mean()),
        "mean_difference": float(treatment.mean() - control.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant_05": bool(p_value < 0.05),
    }


def estimate_sample_size(control_rate: float, target_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    effect_size = proportion_effectsize(control_rate, target_rate)
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(sample_size))


def run_ab_analysis(processed_dir: Path, output_dir: Path) -> dict:
    df = read_table(processed_dir, "ab_test_sample")
    conversion = conversion_z_test(df)
    order_value = order_value_t_test(df)
    sample_size = estimate_sample_size(
        conversion["control_conversion_rate"],
        conversion["control_conversion_rate"] + max(conversion["absolute_lift"], 0.01),
    )

    result = {
        "conversion_test": conversion,
        "order_value_test": order_value,
        "estimated_sample_size_per_group": sample_size,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ab_test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated A/B testing validation.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ab_analysis(args.processed_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
