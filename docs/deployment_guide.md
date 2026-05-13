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
cd ecommerce_growth_platform
```

Upload raw Kaggle CSV files to:

```text
data/raw/
```

The raw files should not be committed to Git.

## Run the Batch Pipeline

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
- For a public demo, restrict access to the Streamlit port through firewall/security group rules if needed.
