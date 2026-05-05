# preprocessing.py
# Cleans and prepares the World Happiness dataset for machine learning.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(df):
    # Task 1: Remove unnecessary columns
    # 'Country', 'Happiness Rank', 'Happiness Level' are not useful as features
    # 'Region' is categorical - we encode it below
    df = df.copy()
    # Rename Job Satisfaction if not already done
    if "Job Satisfaction" in df.columns:
        df = df.rename(columns={"Job Satisfaction": "Job_Satisfaction"})

    # Task 2: Handle missing values - drop rows with any NaN
    print(f"Rows before dropping NaN: {len(df)}")
    df = df.dropna(subset=["Economy", "Family", "Health", "Freedom",
                            "Generosity", "Corruption", "Dystopia",
                            "Job_Satisfaction", "Happiness Score"])
    print(f"Rows after dropping NaN: {len(df)}")

    # Task 3: Encode 'Region' (categorical -> numerical using one-hot encoding)
    df = pd.get_dummies(df, columns=["Region"], drop_first=True)

    # Task 4: Drop non-feature columns
    df = df.drop(columns=["Country", "Happiness Rank"], errors="ignore")
    if "Happiness Level" in df.columns:
        df = df.drop(columns=["Happiness Level"])

    # Task 5: Split into features (X) and target (y)
    # Features: all columns except Happiness Score
    # Target: Happiness Score (continuous -> regression task)
    X = df.drop(columns=["Happiness Score"])
    y = df["Happiness Score"]

    print(f"\nFeatures used: {X.columns.tolist()}")
    print(f"Target: Happiness Score")

    # Task 6: Check final shapes
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # Task 7: Normalize / scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nScaling applied: StandardScaler (mean=0, std=1)")
    print(f"Sample scaled values:\n{X_scaled.head(2)}")

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df = pd.read_csv("data/world_happiness.csv")
    if "Job Satisfaction" in df.columns:
        df = df.rename(columns={"Job Satisfaction": "Job_Satisfaction"})
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print("\nPreprocessing complete.")
