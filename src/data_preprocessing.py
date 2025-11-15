"""
Data preprocessing module for diabetic readmission dataset.

This module contains functions for cleaning, feature engineering, and
preparing the diabetes dataset for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def clean_dataset(df):
    """
    Clean the diabetic readmission dataset.

    Clinical rationale: Remove invalid entries and handle missing values
    appropriately based on clinical context.

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw diabetes dataset

    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()

    print("Starting data cleaning...")
    print(f"Original shape: {df.shape}")

    # Remove encounters with invalid data
    # Race '?' is unknown - can impute or remove
    if 'race' in df_clean.columns:
        df_clean = df_clean[df_clean['race'] != '?']

    # Handle missing values in key clinical variables
    # Replace '?' with NaN for easier handling
    df_clean = df_clean.replace('?', np.nan)

    # Remove columns with >40% missing data
    missing_threshold = 0.4
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

    if cols_to_drop:
        print(f"Dropping columns with >{missing_threshold*100}% missing data: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)

    # Remove duplicate patient encounters (keep most recent)
    if 'patient_nbr' in df_clean.columns and 'encounter_id' in df_clean.columns:
        df_clean = df_clean.sort_values('encounter_id').drop_duplicates(
            subset=['patient_nbr'], keep='last'
        )

    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows")

    return df_clean


def create_target_variable(df):
    """
    Create binary target: readmitted within 30 days (high risk) vs not.

    Clinical context: 30-day readmissions are a key quality metric
    and are often penalized by insurance/Medicare.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'readmitted' column

    Returns:
    --------
    pandas.DataFrame
        Dataset with added 'readmitted_30days' binary target
    """
    if 'readmitted' not in df.columns:
        raise ValueError("Dataset must contain 'readmitted' column")

    df['readmitted_30days'] = (df['readmitted'] == '<30').astype(int)

    print(f"\nTarget variable created:")
    print(f"30-day readmission rate: {df['readmitted_30days'].mean():.2%}")

    return df


def engineer_clinical_features(df):
    """
    Create clinically meaningful features based on domain knowledge.

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset

    Returns:
    --------
    pandas.DataFrame
        Dataset with engineered features
    """
    df_features = df.copy()

    print("\nEngineering clinical features...")

    # 1. Total medications changed (clinical instability indicator)
    med_change_cols = [col for col in df.columns if 'change' in col.lower()]
    if med_change_cols:
        df_features['total_med_changes'] = df_features[med_change_cols].apply(
            lambda x: (x == 'Ch').sum(), axis=1
        )
    else:
        df_features['total_med_changes'] = 0

    # 2. Polypharmacy indicator (high medication burden)
    if 'num_medications' in df_features.columns:
        df_features['polypharmacy'] = (df_features['num_medications'] >= 10).astype(int)

    # 3. Comorbidity score (number of diagnoses)
    if 'number_diagnoses' in df_features.columns:
        df_features['high_comorbidity'] = (df_features['number_diagnoses'] >= 7).astype(int)

    # 4. Long hospital stay
    if 'time_in_hospital' in df_features.columns:
        df_features['long_stay'] = (df_features['time_in_hospital'] > 7).astype(int)

    # 5. High utilization (many procedures/labs)
    if 'num_procedures' in df_features.columns and 'num_lab_procedures' in df_features.columns:
        df_features['high_utilization'] = (
            (df_features['num_procedures'] > 3) |
            (df_features['num_lab_procedures'] > 50)
        ).astype(int)

    # 6. Age groups (clinical risk stratification)
    if 'age' in df_features.columns:
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df_features['age_numeric'] = df_features['age'].map(age_mapping)
        df_features['elderly'] = (df_features['age_numeric'] >= 65).astype(int)

    # 7. Uncontrolled diabetes (no A1C test or abnormal result)
    if 'A1Cresult' in df_features.columns:
        df_features['uncontrolled_diabetes'] = (
            (df_features['A1Cresult'] == '>8') |
            (df_features['A1Cresult'] == 'None')
        ).astype(int)

    # 8. Emergency admission
    if 'admission_type_id' in df_features.columns:
        df_features['emergency_admit'] = (df_features['admission_type_id'] == 1).astype(int)

    # 9. Prior admissions/emergencies
    prior_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    if all(col in df_features.columns for col in prior_cols):
        df_features['prior_utilization'] = (
            df_features['number_outpatient'] +
            df_features['number_emergency'] +
            df_features['number_inpatient']
        )

    print(f"Created {sum([1 for col in df_features.columns if col not in df.columns])} new features")

    return df_features


def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical variables for ML models.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with categorical columns
    categorical_cols : list
        List of categorical column names to encode

    Returns:
    --------
    tuple
        (encoded_dataframe, dict_of_label_encoders)
    """
    df_encoded = df.copy()
    label_encoders = {}

    print(f"\nEncoding {len(categorical_cols)} categorical features...")

    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    return df_encoded, label_encoders


def select_features(df):
    """
    Select final feature set for modeling.

    Based on:
    1. Clinical relevance
    2. Correlation with target
    3. Low multicollinearity

    Parameters:
    -----------
    df : pandas.DataFrame
        Fully processed dataset

    Returns:
    --------
    pandas.DataFrame
        Dataset with selected features only
    """
    feature_cols = [
        # Demographics
        'age_numeric', 'elderly', 'gender',

        # Clinical measures
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_diagnoses',

        # Engineered features
        'polypharmacy', 'high_comorbidity', 'long_stay',
        'high_utilization', 'uncontrolled_diabetes', 'emergency_admit',
        'prior_utilization', 'total_med_changes',

        # Medical history
        'number_inpatient', 'number_emergency', 'number_outpatient',

        # Treatment indicators
        'diabetesMed', 'insulin'
    ]

    # Filter to columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]

    print(f"\nSelected {len(feature_cols)} features for modeling")

    return df[feature_cols]


def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline.

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset

    Returns:
    --------
    tuple
        (X_features, y_target, label_encoders)
    """
    print("="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)

    # Step 1: Clean
    df_clean = clean_dataset(df)

    # Step 2: Create target
    df_clean = create_target_variable(df_clean)

    # Step 3: Engineer features
    df_features = engineer_clinical_features(df_clean)

    # Step 4: Encode categoricals
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    # Exclude target from encoding
    categorical_cols = [col for col in categorical_cols
                       if col not in ['readmitted', 'readmitted_30days']]

    df_encoded, label_encoders = encode_categorical_features(df_features, categorical_cols)

    # Step 5: Select final features
    X = select_features(df_encoded)
    y = df_encoded['readmitted_30days']

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Final dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"30-day readmission rate: {y.mean():.2%}")
    print("="*60)

    return X, y, label_encoders


if __name__ == "__main__":
    # Test the preprocessing pipeline
    import os

    data_path = "../data/raw/diabetic_data.csv"
    if os.path.exists(data_path):
        print("Loading dataset...")
        df_raw = pd.read_csv(data_path)

        X, y, encoders = preprocess_pipeline(df_raw)

        print("\nFeature names:")
        print(X.columns.tolist())

        print("\nSample of processed data:")
        print(X.head())
    else:
        print(f"Dataset not found at {data_path}")
        print("Please run scripts/download_uci_data.py first")
