FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (includes google-cloud-storage for GCS DB sync)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .

# entrypoint.py: downloads signal_history.db from GCS, runs orchestrator,
# uploads updated DB back to GCS.
ENTRYPOINT ["python", "entrypoint.py"]
