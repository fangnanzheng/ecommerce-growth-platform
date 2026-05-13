from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.io import read_table, write_csv


def pick_default_scope(daily_sales: pd.DataFrame) -> tuple[str | None, str]:
    group_columns = ["product_category"]
    if "customer_state" in daily_sales.columns:
        group_columns = ["customer_state", "product_category"]

    top_scope = (
        daily_sales.groupby(group_columns)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .iloc[0]
    )
    state = str(top_scope["customer_state"]) if "customer_state" in top_scope.index else None
    category = str(top_scope["product_category"])
    return state, category


def prepare_series(
    daily_sales: pd.DataFrame,
    category: str | None = None,
    state: str | None = None,
) -> tuple[str | None, str, pd.Series]:
    df = daily_sales.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    if category is None:
        default_state, default_category = pick_default_scope(df)
        category = default_category
        state = state or default_state

    if state and "customer_state" in df.columns:
        df = df[df["customer_state"] == state]

    df = df[df["product_category"] == category]
    if df.empty:
        raise ValueError(f"No daily sales found for state={state or 'ALL'}, category={category}.")

    series = (
        df.groupby("order_date")["revenue"]
        .sum()
        .asfreq("D", fill_value=0)
        .astype(float)
    )
    return state, category, series


def forecast_series(
    series: pd.Series,
    periods: int = 30,
    alpha: float = 0.2,
    method: str = "sarimax",
) -> pd.DataFrame:
    if len(series) < 35:
        raise ValueError("Need at least 35 daily observations for a forecast with confidence intervals.")

    method = method.lower()
    if method not in {"sarimax", "ets", "moving_average"}:
        raise ValueError(f"Unsupported forecast method: {method}")

    z_value = 1.2815515655446004 if abs(alpha - 0.2) < 1e-9 else 1.96

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "sarimax":
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=100)
            forecast = fitted.get_forecast(steps=periods)
            predicted = forecast.predicted_mean
            confidence = forecast.conf_int(alpha=alpha)
            lower = confidence.iloc[:, 0].values
            upper = confidence.iloc[:, 1].values
        elif method == "ets":
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            predicted = fitted.forecast(periods)
            residual_std = float(np.nanstd(fitted.resid, ddof=1))
            horizon_scale = np.sqrt(np.arange(1, periods + 1))
            interval = z_value * residual_std * horizon_scale
            lower = predicted.values - interval
            upper = predicted.values + interval
        else:
            rolling_window = min(28, max(7, len(series) // 8))
            baseline = float(series.tail(rolling_window).mean())
            predicted_index = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=periods, freq="D")
            predicted = pd.Series(np.repeat(baseline, periods), index=predicted_index)
            residual = series - series.rolling(rolling_window, min_periods=7).mean()
            residual_std = float(np.nanstd(residual.dropna(), ddof=1))
            horizon_scale = np.sqrt(np.arange(1, periods + 1))
            interval = z_value * residual_std * horizon_scale
            lower = predicted.values - interval
            upper = predicted.values + interval

    result = pd.DataFrame(
        {
            "order_date": predicted.index,
            "forecast_revenue": np.clip(predicted.values, a_min=0, a_max=None),
            "lower_revenue": np.clip(lower, a_min=0, a_max=None),
            "upper_revenue": np.clip(upper, a_min=0, a_max=None),
        }
    )
    return result


def run_forecast(
    processed_dir: Path,
    output_dir: Path,
    category: str | None = None,
    state: str | None = None,
    periods: int = 30,
    alpha: float = 0.2,
    method: str = "sarimax",
) -> pd.DataFrame:
    try:
        daily_sales = read_table(processed_dir, "state_category_daily_sales")
    except FileNotFoundError:
        daily_sales = read_table(processed_dir, "category_daily_sales")

    state, category, series = prepare_series(daily_sales, category=category, state=state)
    result = forecast_series(series, periods=periods, alpha=alpha, method=method)
    result["customer_state"] = state or "ALL"
    result["product_category"] = category
    result = result[
        [
            "order_date",
            "customer_state",
            "product_category",
            "forecast_revenue",
            "lower_revenue",
            "upper_revenue",
        ]
    ]
    write_csv(result, output_dir, "sales_forecast.csv")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast state/category sales revenue.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--periods", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.2, help="Forecast interval alpha. 0.2 means 80% interval.")
    parser.add_argument(
        "--method",
        choices=["sarimax", "ets", "moving_average"],
        default="sarimax",
        help="Forecast method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecast = run_forecast(
        args.processed_dir,
        args.output_dir,
        args.category,
        args.state,
        args.periods,
        args.alpha,
        args.method,
    )
    print(forecast.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
