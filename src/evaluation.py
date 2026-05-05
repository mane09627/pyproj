# evaluation.py
# Evaluates the trained regression model and saves results.
#
# Evaluation Answers:
# 1. Task: Regression — predicting the Happiness Score (a number)
# 2. Model: Linear Regression
# 3. Performance: Measured by MAE, MSE, and R² Score
# 4. Metric: R² Score tells us what percentage of variance the model explains.
#            MAE tells us the average error in happiness score units.
# 5. Mistakes: With a good R², mistakes should be small (< 0.5 score difference on average)
# 6. Improvements: Could try Decision Tree Regressor, add more features, or tune hyperparameters.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("outputs/results", exist_ok=True)


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model, print and save metrics and predictions."""

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Model Evaluation (Regression) ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error  (MSE): {mse:.4f}")
    print(f"R-squared Score     (R²):  {r2:.4f}")
    print()
    print("Interpretation:")
    print(f"  - On average, the model's predictions are off by {mae:.2f} happiness points.")
    print(f"  - The model explains {r2*100:.1f}% of the variance in Happiness Scores.")
    if r2 > 0.85:
        print("  - This is a strong fit!")
    elif r2 > 0.70:
        print("  - This is a good fit.")
    else:
        print("  - The model could be improved with more complex algorithms.")

    # Save metrics to file
    with open("outputs/results/metrics.txt", "w") as f:
        f.write("World Happiness Regression — Model Evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: Linear Regression\n")
        f.write(f"Task: Regression (predicting Happiness Score)\n\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Mean Squared Error  (MSE): {mse:.4f}\n")
        f.write(f"R-squared Score     (R²):  {r2:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write(f"  - Average prediction error: {mae:.2f} happiness points\n")
        f.write(f"  - Model explains {r2*100:.1f}% of variance in Happiness Scores\n")
        f.write("\nFeatures used:\n")
        for name in feature_names:
            f.write(f"  - {name}\n")
    print("Saved: outputs/results/metrics.txt")

    # Save predictions to CSV
    results_df = pd.DataFrame({
        "Actual_Happiness_Score": y_test.values,
        "Predicted_Happiness_Score": np.round(y_pred, 4),
        "Error": np.round(y_test.values - y_pred, 4)
    })
    results_df.to_csv("outputs/results/predictions.csv", index=False)
    print("Saved: outputs/results/predictions.csv")

    return mae, mse, r2


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from preprocessing import preprocess
    from model import train_model

    df = pd.read_csv("data/world_happiness.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, X_train.columns.tolist())
    print("\nEvaluation complete.")
