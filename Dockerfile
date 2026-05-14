FROM python:3.11-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --prefer-binary \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.13.1 \
    scikit-learn==1.5.2 \
    statsmodels==0.14.4 \
    pyarrow==15.0.2
RUN python -m pip install --no-cache-dir --prefer-binary \
    plotly==5.24.1 \
    openpyxl==3.1.5 \
    pydeck==0.9.1
RUN python -m pip install --no-cache-dir --prefer-binary streamlit==1.57.0

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
