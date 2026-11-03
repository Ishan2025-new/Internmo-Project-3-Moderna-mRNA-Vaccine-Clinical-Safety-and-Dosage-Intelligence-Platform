import os
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

REQUIRED_COLUMNS = [
    "Patient_ID", "Age", "Biomarker_Level", "Pre_Existing_Conditions", "Vaccine_Dose_mcg", "Prior_Reaction_History", 
    "Reaction_Score"
]
NUMERIC_COLS = ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]
CATEGORICAL_COLS = "Prior_Reaction_History"
TARGET_COL = "Reaction_Score"

def validate_and_load_data(path: str) -> pd.DataFrame:
    """
    Load Dataset and validate that all required columns are present (FR-1.1).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input File not found at: {path}")

    df = pd.read_csv(path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Dataset schema validation failed. Missing required columns: {missing_cols}"
        )

    print(f"Dataset successfully loaded. Shape: {df.shape}")
    return df

def clean_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute Missing Values: median for numeric, mode for categorical (FR-1.2).
    """
    df_clean = df.copy()

    # Impute missing numeric columns with median
    for col in NUMERIC_COLS:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Impute missing categorical columns with mode
    if df_clean[CATEGORICAL_COLS].isnull().sum() > 0:
        mode_val = df_clean[CATEGORICAL_COLS].mode()[0]
        df_clean[CATEGORICAL_COLS] = df_clean[CATEGORICAL_COLS].fillna(mode_val)

    # Verify no null values remain
    assert df_clean.isnull().sum().sum() == 0, "Missing Values still exist after cleaning!"
    return df_clean

def prepare_features(df: pd.DataFrame):
    """
    Standardize numeric columns and encode categorical features (FR-1.3).
    """
    df_processed = df.copy()

    scaler = StandardScaler()
    df_processed[NUMERIC_COLS] = scaler.fit_transform(df_processed[NUMERIC_COLS])

    encoder = LabelEncoder()
    df_processed[CATEGORICAL_COLS] = encoder.fit_transform(df_processed[CATEGORICAL_COLS])

    feature_cols = NUMERIC_COLS + [CATEGORICAL_COLS, "Pre_Existing_Conditions"]
    X = df_processed[feature_cols]
    y = df_processed[TARGET_COL]

    return X, y, scaler, encoder

def train_and_evaluate(X, y):
    """
    Split Data, Train Models, Generate Reports and Save Metrices (FR-1.4, FR-1.5, FR-1.6, FR-1.7)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }

    best_model_name = None
    best_f1 = -1.0
    best_model = None
    results = {}

    os.makedirs("outputs/figures", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)

        # Macro average F1 score for comparison
        macro_f1 = report['macro avg']['f1-score']
        results[name] = {
            'model': model,
            'report': report,
            'cm': confusion_matrix(y_test, preds),
            'preds': preds
        }

        print(f"\nModel: {name}".center(20, "="))
        print(classification_report(y_test, preds))

        # Track best model by F1
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            best_model = model

    print(f"\n--> Selected Model: {best_model_name} (Model F1: {best_f1:.4f})")

    # Plot and save confusion matrix of winning model
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        results[best_model_name]['cm'],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=['Low Risk (0)', 'High Risk (1)'],
        yticklabels=['Low Risk (0)', 'High Risk (1)']
    )
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Actual Risk Score")
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png', dpi=150)
    plt.close()

    # Save best model to disk
    joblib.dump(best_model, 'outputs/risk_model.joblib')
    print("Serialized best performing model to 'outputs/risk_model.joblib'")

    return best_model, X_test, y_test

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    input_path = "data/patient_records.csv"
    clean_path = "data/patient_records_clean.csv"

    df_raw = validate_and_load_data(input_path)
    df_clean = clean_missing_data(df_raw)
    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to '{clean_path}'")

    X, y, scaler, encoder = prepare_features(df_clean)
    best_model, X_test, y_test = train_and_evaluate(X, y)