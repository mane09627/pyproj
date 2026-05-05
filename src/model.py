# model.py
# Trains a machine learning model on the World Happiness dataset.
#
# Task type: REGRESSION
# Model: Linear Regression
# Target: Happiness Score (a continuous number)
#
# We use Linear Regression because:
# - The target is a continuous numerical value (Happiness Score)
# - It is interpretable: we can see which features contribute most
# - It is a good baseline for regression problems

import numpy as np
from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """Train a Linear Regression model on training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained: Linear Regression")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")

    # NumPy Task 1: View model coefficients as numpy array
    coef_array = np.array(model.coef_)
    print(f"\nCoefficient array shape: {coef_array.shape}")

    # NumPy Task 2: Find index of the largest coefficient (most influential feature)
    most_important_idx = np.argmax(np.abs(coef_array))
    print(f"Most influential feature index: {most_important_idx}")

    # NumPy Task 3: Compute mean and std of coefficients
    print(f"Coefficients - Mean: {np.mean(coef_array):.4f}, Std: {np.std(coef_array):.4f}")

    # NumPy Task 4: Clip coefficients to [-5, 5] range for inspection (does not change model)
    clipped = np.clip(coef_array, -5, 5)
    print(f"Clipped coefficient range: [{clipped.min():.4f}, {clipped.max():.4f}]")

    return model


if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.insert(0, "src")
    from preprocessing import preprocess

    df = pd.read_csv("data/world_happiness.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    model = train_model(X_train, y_train)
    print("\nModel training complete.")
