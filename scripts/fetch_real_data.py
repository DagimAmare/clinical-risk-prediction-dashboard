"""
Fetch the real Diabetes 130-US Hospitals dataset from UCI ML Repository
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def fetch_dataset():
    """Fetch and save the diabetes dataset from UCI"""

    print("Fetching Diabetes 130-US Hospitals dataset from UCI ML Repository...")

    try:
        # Fetch dataset (ID: 296)
        diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

        # Extract features and targets
        X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
        y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

        # Combine into single dataframe
        df = pd.concat([X, y], axis=1)

        print(f"\nDataset fetched successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")

        # Create directory
        os.makedirs("data/raw", exist_ok=True)

        # Save to CSV
        df.to_csv("data/raw/diabetic_data.csv", index=False)
        print(f"\nSaved to: data/raw/diabetic_data.csv")

        # Display info
        print("\nFirst few rows:")
        print(df.head())

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes.value_counts())

        # Check target variable
        if 'readmitted' in df.columns:
            print("\nReadmission distribution:")
            print(df['readmitted'].value_counts())
            print(f"\nReadmission rate: {(df['readmitted'] != 'NO').mean():.2%}")

        # Save metadata
        metadata = diabetes_130_us_hospitals_for_years_1999_2008.metadata
        print("\nDataset Metadata:")
        print(f"Name: {metadata.get('name', 'N/A')}")
        print(f"Description: {metadata.get('abstract', 'N/A')[:200]}...")

        return df

    except Exception as e:
        print(f"\nError fetching dataset: {e}")
        print("\nPlease check:")
        print("1. Internet connection")
        print("2. UCI ML Repository availability")
        print("3. Dataset ID (296) is correct")
        return None

if __name__ == "__main__":
    df = fetch_dataset()
    if df is not None:
        print("\n✅ Dataset ready for analysis!")
    else:
        print("\n❌ Failed to fetch dataset")
