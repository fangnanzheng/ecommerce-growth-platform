# Deployment Guide

This project can be deployed on a standard VPS such as AWS EC2, Aliyun ECS, or any Linux server with Docker installed.

## Recommended Server

Minimum:

- 2 vCPU
- 4 GB RAM
- 20 GB disk

Comfortable:

- 2-4 vCPU
- 8 GB RAM
- 30+ GB disk

## Server Setup

Install Docker and Docker Compose on the server, then clone the repository:

```bash
git clone <YOUR_REPO_URL>
cd ecommerce-growth-platform
```

Upload raw Kaggle CSV files to:

```text
data/raw/
```

The raw files should not be committed to Git.

## Run the Batch Pipeline

If package downloads are slow, build with a PyPI mirror first:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker compose build dashboard pipeline
```

```bash
docker compose --profile batch run --rm pipeline
```

This generates:

```text
data/processed/
data/output/
```

## Start Dashboard

```bash
docker compose up -d --build dashboard
```

Open:

```text
http://<SERVER_PUBLIC_IP>:8501
```

## Update Deployment

```bash
git pull
docker compose --profile batch run --rm pipeline
docker compose up -d --build dashboard
```

## Stop Services

```bash
docker compose down
```

## Notes

- `data/` is mounted as a volume, so generated files persist outside the Docker image.
- `.dockerignore` excludes raw and generated data from image layers.
- The Docker image intentionally uses the lightweight pandas pipeline dependencies. PySpark remains available for local demos through `requirements-spark.txt` / `environment.yml`.
- For a public demo, restrict access to the Streamlit port through firewall/security group rules if needed.
- On Tencent Cloud or other mainland China servers, Docker Hub and PyPI can be slow or unstable. Configure a Docker registry mirror and use `PIP_INDEX_URL` during image builds.
