"""
Session 04 – Step 1: Data Ingestion
Reads raw IRIS.csv and saves it to the ingested/ folder.
"""

import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE = BASE_DIR / "train.csv"
OUTPUT_FILE = INGESTED_DIR / "spaceship_train.csv"

def ingest_data():
    os.makedirs(INGESTED_DIR, exist_ok=True)
    if not INPUT_FILE.exists():
        print(f"Gagal: File {INPUT_FILE} tidak ditemukan!")
        return

    df = pd.read_csv(INPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data Ingested ke: {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()