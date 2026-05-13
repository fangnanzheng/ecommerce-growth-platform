# Operations Guide

## One-command Pipeline

Run the full local pipeline from the project root:

```bash
python -m src.run_pipeline --engine pandas
```

This executes:

1. ETL from raw Olist CSV files
2. Data quality checks
3. Customer segmentation appendix
4. Repeat purchase propensity model
5. Default SARIMAX forecast
6. Experiment readout
7. Excel report generation

Outputs are written to:

```text
data/processed/
data/output/
```

## Data Quality

Run checks only:

```bash
python -m src.data_pipeline.data_quality
```

Main outputs:

```text
data/output/data_quality_report.json
data/output/data_quality_summary.csv
```

The checks cover:

- required raw files
- raw table row counts
- duplicate primary keys where applicable
- top missing-value rates
- referential integrity across orders, customers, items, payments, and products
- processed table row counts
- fact order payment sanity checks
- repeat-purchase target rate

## Dashboard

Start the app:

```bash
python -m streamlit run app/main.py
```

If a port is occupied, use another port:

```bash
python -m streamlit run app/main.py --server.port=8503
```
