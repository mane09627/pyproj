# main.py
# Runs the full World Happiness ML project pipeline.
# Run from the project root with: python src/main.py
#
# Dataset: World Happiness Report
# Source: https://worldhappiness.report/
# Task: Regression — predicting a country's Happiness Score

import sys
import os

# Add src/ to path so imports work when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd

from data_exploration import load_and_explore
from preprocessing import preprocess
from visualization import create_matplotlib_plots, create_plotly_plots
from model import train_model
from evaluation import evaluate_model


def main():
    print("=" * 60)
    print("  WORLD HAPPINESS REPORT — ML REGRESSION PROJECT")
    print("  Student: Mane Harutyunyan | Grade: 11.5AI")
    print("=" * 60)
    print()

    # Step 1: Load and explore the data
    print(">>> STEP 1: Data Exploration")
    print("-" * 40)
    df = load_and_explore()
    print()

    # Step 2: Create visualizations
    print(">>> STEP 2: Visualizations")
    print("-" * 40)
    raw_df = pd.read_csv("data/world_happiness.csv")
    create_matplotlib_plots(raw_df)
    create_plotly_plots(raw_df)
    print()

    # Step 3: Preprocess the data
    print(">>> STEP 3: Preprocessing")
    print("-" * 40)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print()

    # Step 4: Train the model
    print(">>> STEP 4: Model Training")
    print("-" * 40)
    model = train_model(X_train, y_train)
    print()

    # Step 5: Evaluate the model
    print(">>> STEP 5: Model Evaluation")
    print("-" * 40)
    mae, mse, r2 = evaluate_model(model, X_test, y_test, X_train.columns.tolist())
    print()

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Final R² Score: {r2:.4f}")
    print(f"  Final MAE:      {mae:.4f}")
    print("  Outputs saved in outputs/plots/ and outputs/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
