"""
Run the data preprocessing pipeline and save processed data
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from data_preprocessing import preprocess_pipeline

# Load raw data
print("Loading raw dataset...")
df_raw = pd.read_csv("data/raw/diabetic_data.csv")

# Run preprocessing
X, y, label_encoders = preprocess_pipeline(df_raw)

# Combine for saving
df_processed = X.copy()
df_processed['readmitted_30days'] = y

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# Save processed data
df_processed.to_csv("data/processed/processed_data.csv", index=False)
print(f"\n[OK] Processed data saved to: data/processed/processed_data.csv")

# Save feature names
with open("data/processed/feature_names.txt", 'w') as f:
    f.write('\n'.join(X.columns.tolist()))
print(f"[OK] Feature names saved to: data/processed/feature_names.txt")

# Save label encoders
import joblib
joblib.dump(label_encoders, "data/processed/label_encoders.pkl")
print(f"[OK] Label encoders saved to: data/processed/label_encoders.pkl")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE - DATA READY FOR MODEL TRAINING")
print("="*60)
