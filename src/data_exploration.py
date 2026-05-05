# data_exploration.py
# Dataset: World Happiness Report
# Source: Kaggle / World Happiness Report (https://worldhappiness.report/)
#
# Column descriptions:
#   Country          - Name of the country
#   Happiness Rank   - Global rank by happiness score
#   Happiness Score  - Overall happiness score (target variable)
#   Economy          - GDP per capita contribution to happiness
#   Family           - Social support contribution
#   Health           - Life expectancy contribution
#   Freedom          - Freedom to make life choices contribution
#   Generosity       - Generosity contribution
#   Corruption       - Perception of corruption (lower = more corrupt)
#   Dystopia         - Hypothetical baseline reference value
#   Job Satisfaction - % of people satisfied with their job
#   Region           - World region the country belongs to
#
# Task type: REGRESSION
# Target column: Happiness Score (a continuous numerical value)

import pandas as pd
import numpy as np


def load_and_explore():
    # Task 1: Load the dataset
    df = pd.read_csv("data/world_happiness.csv")

    # Task 2: Display first rows
    print("=== First 5 Rows ===")
    print(df.head())
    print()

    # Task 3: Show dataset information
    print("=== Dataset Info ===")
    df.info()
    print()

    # Task 4: Summary statistics
    print("=== Summary Statistics ===")
    print(df.describe())
    print()

    # Task 5: Check for missing values
    print("=== Missing Values ===")
    print(df.isnull().sum())
    print()

    # Task 7: Select specific columns (numerical features only)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("=== Numerical Columns ===")
    print(numerical_cols)
    print()

    # Task 8: Filter rows - countries with Happiness Score above 6.0
    happy_countries = df[df["Happiness Score"] > 6.0]
    print(f"=== Countries with Happiness Score > 6.0: {len(happy_countries)} ===")
    print(happy_countries[["Country", "Happiness Score"]].head(10))
    print()

    # Task 9: Sort by Happiness Score descending
    sorted_df = df.sort_values("Happiness Score", ascending=False)
    print("=== Top 10 Happiest Countries ===")
    print(sorted_df[["Country", "Happiness Score", "Region"]].head(10))
    print()

    # Task 10: Group by Region and compute mean happiness
    print("=== Mean Happiness Score by Region ===")
    region_means = df.groupby("Region")["Happiness Score"].mean().sort_values(ascending=False)
    print(region_means)
    print()

    # Task 11: Create a new column - Happiness Level category
    df["Happiness Level"] = pd.cut(
        df["Happiness Score"],
        bins=[0, 4.5, 6.0, 10],
        labels=["Low", "Medium", "High"]
    )
    print("=== Happiness Level Distribution ===")
    print(df["Happiness Level"].value_counts())
    print()

    # Task 12: Rename column for cleaner usage
    df = df.rename(columns={"Job Satisfaction": "Job_Satisfaction"})
    print("=== Columns after renaming ===")
    print(df.columns.tolist())
    print()

    return df


if __name__ == "__main__":
    df = load_and_explore()
    print("Data exploration complete.")
