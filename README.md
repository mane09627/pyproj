# World Happiness Report — ML Regression Project

**Student:** Mane Harutyunyan  
**Grade:** 11.5AI

---

## Dataset

- **Source:** [World Happiness Report](https://worldhappiness.report/)
- **File:** `data/world_happiness.csv`
- **Rows:** 153 countries | **Columns:** 12

### Column Descriptions

| Column | Description |
|---|---|
| Country | Country name |
| Happiness Rank | Global rank by happiness score |
| Happiness Score | Overall happiness (target variable) |
| Economy | GDP per capita contribution |
| Family | Social support contribution |
| Health | Life expectancy contribution |
| Freedom | Freedom to make life choices |
| Generosity | Generosity contribution |
| Corruption | Perception of government corruption |
| Dystopia | Baseline dystopia residual |
| Job Satisfaction | % of people satisfied with their job |
| Region | World region |

---

## Task

**Type:** Regression  
**Target:** `Happiness Score` (a continuous number between ~3 and ~8)  
**Goal:** Predict a country's happiness score from economic and social indicators.

---

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## Project Structure

```
my_ml_project/
├── data/
│   └── world_happiness.csv
├── src/
│   ├── data_exploration.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   ├── evaluation.py
│   └── main.py
├── outputs/
│   ├── plots/
│   │   ├── plot_1_happiness_by_region.png
│   │   ├── plot_2_happiness_distribution.png
│   │   ├── plot_3_economy_vs_happiness.png
│   │   ├── plot_4_health_vs_happiness.html
│   │   └── plot_5_top20_countries.html
│   └── results/
│       ├── metrics.txt
│       └── predictions.csv
└── requirements.txt
```

---

## Model Evaluation Summary

- **Model:** Linear Regression
- **Metrics:** MAE, MSE, R² Score
- **What R² means:** How much of the variation in happiness the model can explain.
- **Possible improvements:** Try Decision Tree Regressor, add interaction features, or collect more data per country over time.
