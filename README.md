# E-Commerce Growth & Experimentation Platform

An end-to-end analytics product built on the Olist Brazilian E-Commerce Dataset. The project demonstrates how ambiguous marketplace growth goals can be translated into a reproducible data pipeline, growth dashboard, repeat-purchase model, demand forecast, experiment readout, automated Excel report, and Dockerized deployment path.

## What This Project Shows

- **Data engineering:** raw CSV ingestion, feature tables, data quality checks, and one-command pipeline orchestration.
- **Growth analytics:** revenue trends, state/category mix, delivery friction, and repeat-purchase behavior.
- **Machine learning:** first-order repeat purchase propensity model with feature drivers and top-k lift analysis.
- **Forecasting:** state/category revenue forecasts with SARIMAX, ETS, and moving-average options.
- **Experimentation:** deterministic demo split, conversion Z-test, order-value Welch T-test, p-values, and sample-size planning.
- **BI delivery:** Streamlit dashboard and automated Excel management report.
- **Deployment:** Docker and Docker Compose for local or VPS deployment.

## Dataset

This project uses the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Raw data is not committed to this repository. Download the CSV files from Kaggle and place them in:

```text
data/raw/
```

Required files:

```text
olist_customers_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_products_dataset.csv
product_category_name_translation.csv
```

Optional Olist files can also be placed in `data/raw/`; they are ignored by Git.

## Quick Start

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ecommerce-growth
```

Validate the default local setup:

```bash
python -m src.utils.validate_setup
```

Validate optional Spark support:

```bash
python -m src.utils.validate_setup --with-spark
```

Run the full pipeline:

```bash
python -m src.run_pipeline --engine pandas
```

Start the dashboard:

```bash
python -m streamlit run app/main.py
```

Open:

```text
http://localhost:8501
```

## Docker

The Docker image uses the lightweight pandas pipeline by default. PySpark support remains available for local Spark demos through `requirements-spark.txt` / `environment.yml`, but it is intentionally not installed in the default dashboard image to keep VPS deployment fast and stable.

Build and start the dashboard:

```bash
docker compose up --build dashboard
```

Run the batch pipeline in Docker:

```bash
docker compose --profile batch run --rm pipeline
```

The `data/` folder is mounted as a volume, so raw data and generated outputs stay outside the image.

## Main Commands

```bash
# Full product pipeline
python -m src.run_pipeline --engine pandas

# Data quality only
python -m src.data_pipeline.data_quality

# Repeat-purchase model only
python -m src.models.repeat_purchase_predict

# Forecast only
python -m src.models.forecasting --method sarimax --periods 30

# Experiment readout only
python -m src.experiments.ab_testing

# Excel report only
python -m src.utils.excel_generator
```

## Project Structure

```text
ecommerce-growth-platform/
|-- app/                         # Streamlit dashboard
|-- data/
|   |-- raw/                     # Kaggle CSV files, not committed
|   |-- processed/               # generated feature tables, not committed
|   `-- output/                  # generated reports/model outputs, not committed
|-- docs/                        # setup, operations, deployment notes
|-- notebooks/                   # optional exploration
|-- src/
|   |-- data_pipeline/           # ETL, SQL logic, data quality checks
|   |-- experiments/             # A/B test validation
|   |-- models/                  # repeat model, forecasting, segmentation appendix
|   `-- utils/                   # IO, setup validation, Excel report
|-- Dockerfile
|-- docker-compose.yml
|-- environment.yml
|-- requirements-spark.txt       # optional Spark support
`-- requirements.txt             # lightweight Docker/pandas runtime
```

## Notes

- The default public workflow uses `--engine pandas` because the Olist dataset is small enough for fast local processing.
- PySpark support is included to demonstrate scalable data engineering architecture; install `requirements-spark.txt` or use `environment.yml` when Spark is needed.
- On Windows, local Spark writes may require `HADOOP_HOME/bin/winutils.exe`; pandas mode avoids that friction.
- Generated files under `data/processed/` and `data/output/` are intentionally ignored.

## License and Data Usage

The Olist dataset is licensed separately by its Kaggle publisher. This repository does not include raw dataset files. Follow the dataset license and Kaggle terms when using the data.
