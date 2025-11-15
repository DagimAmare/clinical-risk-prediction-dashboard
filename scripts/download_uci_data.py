"""
Download the real Diabetes 130-US Hospitals dataset directly from UCI
"""

import requests
import zipfile
import os
import pandas as pd

def download_dataset():
    """Download and extract the diabetes dataset from UCI"""

    # Create directory
    os.makedirs("data/raw", exist_ok=True)

    # Direct download link
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    zip_path = "data/raw/dataset_diabetes.zip"

    print("Downloading Diabetes 130-US Hospitals dataset from UCI...")
    print(f"URL: {url}")

    try:
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')

        print("\n\nDownload complete!")

        # Extract ZIP file
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw/")

        print("Extraction complete!")

        # Load and display dataset info
        csv_path = "data/raw/diabetic_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\nDataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Rows: {df.shape[0]:,}")
            print(f"Columns: {df.shape[1]}")

            print("\nColumn names:")
            print(df.columns.tolist())

            if 'readmitted' in df.columns:
                print("\nReadmission distribution:")
                print(df['readmitted'].value_counts())
                readmit_rate = (df['readmitted'] != 'NO').mean()
                readmit_30_rate = (df['readmitted'] == '<30').mean()
                print(f"\nTotal readmission rate: {readmit_rate:.2%}")
                print(f"30-day readmission rate: {readmit_30_rate:.2%}")

            print("\nFirst few rows:")
            print(df.head())

            print("\nDataset is ready for preprocessing!")
            return df
        else:
            print(f"Error: Could not find diabetic_data.csv in extracted files")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\nError downloading dataset: {e}")
        return None
    except zipfile.BadZipFile as e:
        print(f"\nError extracting zip file: {e}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None

if __name__ == "__main__":
    df = download_dataset()
    if df is not None:
        print("\nSuccess! Dataset ready for analysis.")
    else:
        print("\nFailed to download dataset.")
