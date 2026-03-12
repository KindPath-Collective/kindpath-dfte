"""
DFTE Cloud Run Job — Entrypoint
================================
Wraps orchestrator.py with GCS-backed SQLite persistence.

Flow:
  1. Download signal_history.db from GCS (if it exists)
  2. Set SIGNAL_DB_PATH to the local copy
  3. Run orchestrator.py with any CLI args passed to this script
  4. Upload the updated DB back to GCS (always, even on error)

Without this wrapper the SQLite DB would be lost when the Cloud Run
task container exits. With it, every run reads and writes to the same
durable GCS object.

Environment variables:
  SIGNAL_BUCKET  — GCS bucket name (default: kindpath-bmr-signals)
  SIGNAL_DB_PATH — overrides local DB path (default: /tmp/signal_history.db)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("entrypoint")

GCS_BUCKET    = os.environ.get("SIGNAL_BUCKET", "kindpath-bmr-signals")
GCS_DB_OBJECT = "signal_history.db"
LOCAL_DB_PATH = os.environ.get("SIGNAL_DB_PATH", "/tmp/signal_history.db")
GCS_URI       = f"gs://{GCS_BUCKET}/{GCS_DB_OBJECT}"


def _gcs_client():
    from google.cloud import storage  # type: ignore
    return storage.Client()


def download_db() -> None:
    try:
        client = _gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(GCS_DB_OBJECT)
        if blob.exists():
            blob.download_to_filename(LOCAL_DB_PATH)
            size = os.path.getsize(LOCAL_DB_PATH)
            log.info(f"Downloaded {GCS_URI} → {LOCAL_DB_PATH} ({size:,} bytes)")
        else:
            log.info(f"{GCS_URI} not found — fresh database will be created this run")
    except Exception as exc:
        log.warning(f"GCS download skipped ({exc}) — using fresh or existing local DB")


def upload_db() -> None:
    if not os.path.exists(LOCAL_DB_PATH):
        log.warning("No local DB file to upload — skipping GCS upload")
        return
    try:
        client = _gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(GCS_DB_OBJECT)
        blob.upload_from_filename(LOCAL_DB_PATH)
        size = os.path.getsize(LOCAL_DB_PATH)
        log.info(f"Uploaded {LOCAL_DB_PATH} → {GCS_URI} ({size:,} bytes)")
    except Exception as exc:
        log.error(f"GCS upload FAILED — signal history may be lost: {exc}")


def main() -> int:
    # ── 1. Pull persistent DB ────────────────────────────────────────────────
    download_db()

    # ── 2. Point SignalLogger at the local path ──────────────────────────────
    os.environ["SIGNAL_DB_PATH"] = LOCAL_DB_PATH

    # ── 3. Run orchestrator with any forwarded args ──────────────────────────
    cmd = [sys.executable, "orchestrator.py"] + sys.argv[1:]
    log.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=os.environ.copy())

    # ── 4. Push updated DB back to GCS (always — even on non-zero exit) ──────
    upload_db()

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
