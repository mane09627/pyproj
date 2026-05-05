# visualization.py
# Creates and saves Matplotlib visualizations + HTML plot files.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import os

os.makedirs("outputs/plots", exist_ok=True)


def create_matplotlib_plots(df):
    df = df.copy()
    if "Job Satisfaction" in df.columns:
        df = df.rename(columns={"Job Satisfaction": "Job_Satisfaction"})

    # Plot 1: Bar chart - Mean Happiness Score by Region
    region_means = df.groupby("Region")["Happiness Score"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(region_means)))
    ax.bar(region_means.index, region_means.values, color=colors, edgecolor="white")
    ax.set_title("Mean Happiness Score by World Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Mean Happiness Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_1_happiness_by_region.png", dpi=150)
    plt.close()
    print("Saved: plot_1_happiness_by_region.png")

    # Plot 2: Histogram - Distribution of Happiness Scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Happiness Score"], bins=20, color="coral", edgecolor="white")
    ax.set_title("Distribution of Happiness Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("Happiness Score", fontsize=12)
    ax.set_ylabel("Number of Countries", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_2_happiness_distribution.png", dpi=150)
    plt.close()
    print("Saved: plot_2_happiness_distribution.png")

    # Plot 3: Scatter - Economy vs Happiness Score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Economy"], df["Happiness Score"], alpha=0.7,
               color="mediumseagreen", edgecolors="white", s=60)
    ax.set_title("Economy (GDP per capita) vs Happiness Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Economy (GDP per capita contribution)", fontsize=12)
    ax.set_ylabel("Happiness Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_3_economy_vs_happiness.png", dpi=150)
    plt.close()
    print("Saved: plot_3_economy_vs_happiness.png")


def _png_to_html(png_path, title, subtitle):
    with open(png_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body style="font-family:sans-serif;background:#f9f9f9;text-align:center;padding:20px">
  <h2>{title}</h2>
  <p>{subtitle}</p>
  <img src="data:image/png;base64,{img_data}" style="max-width:1000px;border:1px solid #ccc;border-radius:8px"/>
  <p style="color:#888;font-size:12px">World Happiness Report — Mane Harutyunyan, 11.5AI</p>
</body>
</html>"""


def create_plotly_plots(df):
    df = df.copy()
    if "Job Satisfaction" in df.columns:
        df = df.rename(columns={"Job Satisfaction": "Job_Satisfaction"})

    # HTML Plot 1: Health vs Happiness by Region
    fig, ax = plt.subplots(figsize=(10, 6))
    regions = df["Region"].unique()
    for region in regions:
        subset = df[df["Region"] == region]
        ax.scatter(subset["Health"], subset["Happiness Score"], label=region, alpha=0.8, s=60)
    ax.set_title("Health vs Happiness Score by Region", fontsize=13, fontweight="bold")
    ax.set_xlabel("Health / Life Expectancy Contribution", fontsize=11)
    ax.set_ylabel("Happiness Score", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    tmp = "outputs/plots/_tmp4.png"
    plt.savefig(tmp, dpi=120)
    plt.close()
    html = _png_to_html(tmp, "Health vs Happiness Score by Region",
                        "Each dot represents a country, colored by region.")
    with open("outputs/plots/plot_4_health_vs_happiness.html", "w") as f:
        f.write(html)
    os.remove(tmp)
    print("Saved: plot_4_health_vs_happiness.html")

    # HTML Plot 2: Top 20 Countries
    top20 = df.nlargest(20, "Happiness Score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [plt.cm.tab20(i / 20) for i in range(len(top20))]
    ax.bar(top20["Country"], top20["Happiness Score"], color=colors, edgecolor="white")
    ax.set_title("Top 20 Happiest Countries", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Happiness Score", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    tmp = "outputs/plots/_tmp5.png"
    plt.savefig(tmp, dpi=120)
    plt.close()
    html = _png_to_html(tmp, "Top 20 Happiest Countries",
                        "Countries with the highest happiness scores in the dataset.")
    with open("outputs/plots/plot_5_top20_countries.html", "w") as f:
        f.write(html)
    os.remove(tmp)
    print("Saved: plot_5_top20_countries.html")


if __name__ == "__main__":
    df = pd.read_csv("data/world_happiness.csv")
    create_matplotlib_plots(df)
    create_plotly_plots(df)
    print("\nAll visualizations saved.")
