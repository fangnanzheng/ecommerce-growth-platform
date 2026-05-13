# Local Setup Guide

This guide is for running the project from a clean local Python environment.

## Why Not Use Base Conda?

You can run small scripts from `base`, but it is better to create a dedicated environment for this project. A project environment avoids dependency conflicts across unrelated projects.

Create the environment:

```bash
conda env create -f environment.yml
conda activate ecommerce-growth
```

Update it later if dependencies change:

```bash
conda env update -f environment.yml --prune
conda activate ecommerce-growth
```

## Validate Setup

From the project root:

```bash
python -m src.utils.validate_setup
```

This checks Python packages, Java/PySpark availability, and required raw data files.

## `python file.py` vs `python -m package.module`

Use module execution from the project root:

```bash
python -m src.models.repeat_purchase_predict
```

This treats `src` as a Python package and makes imports more reliable than running files directly.

## Recommended Local Workflow

```bash
python -m src.utils.validate_setup
python -m src.run_pipeline --engine pandas
python -m streamlit run app/main.py
```

## Spark Notes

The project includes PySpark support, but Olist is small enough for pandas mode. On Windows, local Spark writes may require `HADOOP_HOME/bin/winutils.exe`; use pandas mode unless Spark is specifically needed.
