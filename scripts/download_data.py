"""
Script to download the Diabetes 130-US Hospitals dataset from UCI ML Repository
"""

import urllib.request
import zipfile
import os
import pandas as pd

def download_dataset():
    """Download and extract the diabetes dataset"""

    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)

    # Try multiple sources for the dataset
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
        "https://archive.ics.uci.edu/static/public/296/diabetes+130+us+hospitals+for+years+1999+2008.zip"
    ]

    downloaded = False
    for url in urls:
        try:
            print(f"Attempting to download from: {url}")
            urllib.request.urlretrieve(url, "data/raw/dataset_diabetes.zip")
            print("Download successful!")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if not downloaded:
        print("\nCouldn't download from UCI repository.")
        print("Dataset may need to be downloaded manually from:")
        print("https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008")
        return False

    # Extract the zip file
    try:
        print("\nExtracting files...")
        with zipfile.ZipFile("data/raw/dataset_diabetes.zip", 'r') as zip_ref:
            zip_ref.extractall("data/raw/")
        print("Extraction complete!")

        # Try to load and display dataset info
        csv_files = [f for f in os.listdir("data/raw") if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(f"data/raw/{csv_files[0]}")
            print(f"\nDataset shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())

        return True

    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
