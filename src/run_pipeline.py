from __future__ import annotations

import argparse
from pathlib import Path

from src.data_pipeline.data_quality import run_data_quality_checks
from src.data_pipeline.etl_pyspark import run_pandas_pipeline, run_spark_pipeline, spark_runtime_available
from src.experiments.ab_testing import run_ab_analysis
from src.models.forecasting import run_forecast
from src.models.repeat_purchase_predict import run_repeat_purchase_model
from src.models.segmentation import run_segmentation
from src.utils.excel_generator import generate_excel_report


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    engine: str = "auto",
    skip_segmentation: bool = False,
) -> None:
    print("[1/7] Building processed data")
    if engine == "spark":
        run_spark_pipeline(raw_dir, processed_dir)
    elif engine == "pandas":
        run_pandas_pipeline(raw_dir, processed_dir)
    elif spark_runtime_available():
        run_spark_pipeline(raw_dir, processed_dir)
    else:
        print("Spark runtime is not fully available. Using pandas ETL.")
        run_pandas_pipeline(raw_dir, processed_dir)

    print("[2/7] Running data quality checks")
    run_data_quality_checks(raw_dir, processed_dir, output_dir)

    if skip_segmentation:
        print("[3/7] Skipping segmentation")
    else:
        print("[3/7] Running customer segmentation appendix")
        run_segmentation(processed_dir, output_dir)

    print("[4/7] Training repeat purchase model")
    run_repeat_purchase_model(processed_dir, output_dir)

    print("[5/7] Building default forecast")
    run_forecast(processed_dir, output_dir, method="sarimax")

    print("[6/7] Running experiment readout")
    run_ab_analysis(processed_dir, output_dir)

    print("[7/7] Generating Excel report")
    generate_excel_report(output_dir, output_dir / "ecommerce_growth_report.xlsx")

    print("Pipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full E-Commerce Growth Platform pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--engine", choices=["auto", "pandas", "spark"], default="auto")
    parser.add_argument("--skip-segmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        engine=args.engine,
        skip_segmentation=args.skip_segmentation,
    )


if __name__ == "__main__":
    main()
