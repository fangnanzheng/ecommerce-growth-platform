from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from src.data_pipeline.etl_pyspark import RAW_TABLES


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"

PYTHON_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyspark": "pyspark",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "streamlit": "streamlit",
    "plotly": "plotly",
    "openpyxl": "openpyxl",
    "pyarrow": "pyarrow",
}


def check_python_version() -> list[str]:
    messages = []
    version = sys.version_info
    if version.major == 3 and version.minor in {10, 11, 12}:
        messages.append(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    else:
        messages.append(
            f"[WARN] Python {version.major}.{version.minor}.{version.micro}; recommended: Python 3.11"
        )
    return messages


def check_packages() -> list[str]:
    messages = []
    for import_name, package_name in PYTHON_PACKAGES.items():
        if importlib.util.find_spec(import_name):
            messages.append(f"[OK] Python package installed: {package_name}")
        else:
            messages.append(f"[MISSING] Python package missing: {package_name}")
    return messages


def check_java() -> list[str]:
    messages = []
    if not shutil.which("java"):
        return ["[MISSING] Java was not found on PATH; PySpark needs Java 17."]

    result = subprocess.run(["java", "-version"], capture_output=True, text=True, check=False)
    version_text = result.stderr.splitlines()[0] if result.stderr else "java found"
    messages.append(f"[OK] {version_text}")

    if platform.system().lower() == "windows":
        hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
        winutils_path = Path(hadoop_home) / "bin" / "winutils.exe" if hadoop_home else None
        if winutils_path and winutils_path.exists():
            messages.append(f"[OK] Windows Spark helper found: {winutils_path}")
        else:
            messages.append(
                "[WARN] Windows local Spark writes need HADOOP_HOME with bin/winutils.exe. "
                "Use --engine pandas until this is configured."
            )

    return messages


def check_raw_files() -> list[str]:
    messages = []
    for filename in RAW_TABLES.values():
        path = RAW_DIR / filename
        if path.exists():
            messages.append(f"[OK] Raw data file found: {filename}")
        else:
            messages.append(f"[MISSING] Raw data file missing: {filename}")
    return messages


def main() -> None:
    sections = {
        "Python": check_python_version(),
        "Packages": check_packages(),
        "Java": check_java(),
        "Raw data": check_raw_files(),
    }

    for section, messages in sections.items():
        print(f"\n{section}")
        print("-" * len(section))
        for message in messages:
            print(message)


if __name__ == "__main__":
    main()
