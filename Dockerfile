FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .env .

# Expose ports
# 7860 - Gradio app
# 8000 - Prometheus Python metrics
# 9100 - Prometheus node exporter
EXPOSE 7860 8000 9100

# Start node exporter in background and run the app
CMD prometheus-node-exporter & python app.py