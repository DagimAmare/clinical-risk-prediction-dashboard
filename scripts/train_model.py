"""
Train the XGBoost model for 30-day readmission prediction
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from sklearn.model_selection import train_test_split
from model import ReadmissionPredictor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations/static_plots", exist_ok=True)

# Load processed data
print("Loading processed dataset...")
df = pd.read_csv("data/processed/processed_data.csv")

# Split features and target
X = df.drop('readmitted_30days', axis=1)
y = df['readmitted_30days']

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"30-day readmission rate: {y.mean():.2%}")

# Train-test split (80-20)
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training readmission rate: {y_train.mean():.2%}")
print(f"Test readmission rate: {y_test.mean():.2%}")

# Train XGBoost model
print("\n" + "="*60)
print("TRAINING XGBOOST MODEL")
print("="*60)

model = ReadmissionPredictor(model_type='xgboost')
model.train(X_train, y_train, handle_imbalance=True)

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING MODEL ON TEST SET")
print("="*60)

results = model.evaluate(X_test, y_test)

# Save evaluation metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'ROC-AUC', 'F1 Score', 'Sensitivity', 'Specificity'],
    'Value': [
        results['accuracy'],
        results['roc_auc'],
        results['f1'],
        results['sensitivity'],
        results['specificity']
    ]
})
metrics_df.to_csv("visualizations/static_plots/model_metrics.csv", index=False)
print("\n[OK] Metrics saved to: visualizations/static_plots/model_metrics.csv")

# Generate visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

print("\n1. Confusion Matrix...")
model.plot_confusion_matrix(
    y_test,
    results['predictions'],
    save_path='visualizations/static_plots/confusion_matrix.png'
)

print("\n2. ROC Curve...")
model.plot_roc_curve(
    y_test,
    results['probabilities'],
    save_path='visualizations/static_plots/roc_curve.png'
)

print("\n3. Feature Importance...")
model.plot_feature_importance(
    top_n=15,
    save_path='visualizations/static_plots/feature_importance.png'
)

# Save the trained model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model.save_model('models/readmission_model.pkl')

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print("\nModel Performance Summary:")
print(f"  - Accuracy:    {results['accuracy']:.1%}")
print(f"  - ROC-AUC:     {results['roc_auc']:.4f}")
print(f"  - F1 Score:    {results['f1']:.4f}")
print(f"  - Sensitivity: {results['sensitivity']:.4f}")
print(f"  - Specificity: {results['specificity']:.4f}")
print("\nFiles created:")
print("  - models/readmission_model.pkl")
print("  - visualizations/static_plots/confusion_matrix.png")
print("  - visualizations/static_plots/roc_curve.png")
print("  - visualizations/static_plots/feature_importance.png")
print("  - visualizations/static_plots/model_metrics.csv")
print("="*60)
