"""
Machine learning model module for 30-day hospital readmission prediction.

This module contains the ReadmissionPredictor class for training and
evaluating XGBoost models on the diabetes readmission dataset.
"""

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class ReadmissionPredictor:
    """
    Machine learning model for predicting 30-day hospital readmission.

    This class encapsulates the entire ML pipeline for readmission prediction,
    including training, evaluation, and visualization.
    """

    def __init__(self, model_type='xgboost'):
        """
        Initialize the readmission predictor.

        Parameters:
        -----------
        model_type : str
            Type of model to use ('xgboost', 'random_forest', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def train(self, X_train, y_train, handle_imbalance=True):
        """
        Train the readmission prediction model.

        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.array
            Training features
        y_train : pandas.Series or numpy.array
            Training labels
        handle_imbalance : bool
            Whether to use SMOTE to balance classes
        """
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Training set size: {X_train.shape}")
        print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")

        # Handle class imbalance with SMOTE
        if handle_imbalance:
            print("\nApplying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {X_train_balanced.shape}")
            print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Select and train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train the model
        self.model.fit(X_train_balanced, y_train_balanced)

        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        print(f"\n[OK] {self.model_type.upper()} model trained successfully!")

    def predict(self, X):
        """
        Predict readmission (0 or 1).

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Features to predict

        Returns:
        --------
        numpy.array
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict readmission probability.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Features to predict

        Returns:
        --------
        numpy.array
            Probability of readmission (class 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation.

        Parameters:
        -----------
        X_test : pandas.DataFrame or numpy.array
            Test features
        y_test : pandas.Series or numpy.array
            Test labels

        Returns:
        --------
        dict
            Dictionary containing predictions, probabilities, and metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)

        # Generate predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Not Readmitted', 'Readmitted']))

        # Print key metrics
        print(f"\n{'='*60}")
        print(f"ACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"F1 SCORE:  {f1:.4f}")
        print(f"{'='*60}")

        # Calculate confusion matrix values
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f"\nSensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity:          {specificity:.4f}")

        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix.

        Parameters:
        -----------
        y_test : array-like
            True labels
        y_pred : array-like
            Predicted labels
        save_path : str, optional
            Path to save the plot
        """
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Readmitted', 'Readmitted'],
                   yticklabels=['Not Readmitted', 'Readmitted'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - 30-Day Readmission Prediction',
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot ROC curve.

        Parameters:
        -----------
        y_test : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        save_path : str, optional
            Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Readmission Risk Prediction',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] ROC curve saved to: {save_path}")

        plt.show()

    def plot_feature_importance(self, top_n=15, save_path=None):
        """
        Plot feature importance.

        Parameters:
        -----------
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save the plot
        """
        if self.model_type not in ['xgboost', 'random_forest']:
            print(f"Feature importance not available for {self.model_type}")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        plt.barh(range(top_n), importances[indices], color=colors)
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Clinical Features for Readmission Prediction',
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Feature importance plot saved to: {save_path}")

        plt.show()

        # Print feature importances
        print("\nTop Feature Importances:")
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {self.feature_names[idx]:25s}: {importances[idx]:.4f}")

    def save_model(self, filepath):
        """
        Save trained model to file.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self, filepath)
        print(f"\n[OK] Model saved to: {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Load trained model from file.

        Parameters:
        -----------
        filepath : str
            Path to the saved model

        Returns:
        --------
        ReadmissionPredictor
            Loaded model instance
        """
        model = joblib.load(filepath)
        print(f"[OK] Model loaded from: {filepath}")
        return model


if __name__ == "__main__":
    print("ReadmissionPredictor Model Module")
    print("This module provides ML models for 30-day readmission prediction")
    print("\nUsage:")
    print("  from model import ReadmissionPredictor")
    print("  model = ReadmissionPredictor(model_type='xgboost')")
    print("  model.train(X_train, y_train)")
    print("  results = model.evaluate(X_test, y_test)")
