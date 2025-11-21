# Dockerfile for Case Study 4 – GCP Cloud Run

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Defaults (Cloud Run will override PORT)
ENV PORT=8080
ENV APP_MODE=api

CMD ["python", "app.py"]
