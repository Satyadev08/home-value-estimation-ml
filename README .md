# Home Value Estimation with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A comparative study of machine learning models for estimating residential property values using the Zillow dataset. Five regression models are benchmarked — Linear Regression, XGBoost, Lasso, Random Forest, and Ridge — evaluated across MAE, RMSE, MAPE, and R² metrics.

---

## 📋 Overview

This project builds a data pipeline and model comparison framework for predicting home values (Zillow's log-error target). The focus is on understanding how different ML approaches handle heterogeneous real estate data including property features, market trends, and neighborhood attributes.

**Best Performing Models:** XGBoost & Random Forest
**Key Finding:** XGBoost achieves the best balance of training and validation accuracy; Linear and Ridge regression overfit severely on this dataset.

---

## 🗂️ Project Structure

```
.
├── home_value_estimation.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
├── requirements.txt              # Python dependencies
└── README.md
```

> **Dataset:** `Zillow.csv` — Zillow property records with structured features (size, bedrooms, bathrooms, year built, tax data, location) and a log-error regression target. Not included in the repo due to size; see Data section below.

---

## 🔄 Pipeline

```
Zillow.csv
    → Drop columns: >60% missing or zero variance
    → Impute: mode (categorical), mean (numerical)
    → Outlier removal: target ∈ (−1, 1)
    → Label encode categorical features
    → Drop highly correlated features (|r| > 0.8)
    → StandardScaler normalization
    → Train/Val split (90/10, random_state=22)
    → Train 5 models
    → Evaluate: MAE, RMSE, MAPE, R²
```

---

## 🧠 Models Compared

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Baseline | Overfits severely |
| **XGBoost** | Gradient Boosted Trees | ✅ Best overall |
| Lasso | Regularized Linear | Poor — feature selection too aggressive |
| **Random Forest** | Ensemble | ✅ Strong generalization |
| Ridge | Regularized Linear | Overfits |

---

## 📊 Results Summary

| Model | Train MAE | Val MAE | Val R² |
|-------|-----------|---------|--------|
| Linear Regression | ~0.0 | ~0.0 | ~1.0 (overfit) |
| **XGBoost** | 0.0016 | **0.0058** | **~0.88** |
| Lasso | 0.0707 | 0.0706 | low |
| **Random Forest** | 0.0001 | **0.0002** | **~0.99** |
| Ridge | ~0.0 | ~0.0 | ~1.0 (overfit) |

> Linear and Ridge show near-zero error on training due to feature leakage — they do not generalize. XGBoost is the recommended production model.

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/home-value-estimation-ml.git
cd home-value-estimation-ml
pip install -r requirements.txt
```

### Dataset

Download the Zillow dataset and place it as `Zillow.csv` in the project root. Sources:
- [Zillow Prize: Zillow's Home Value Prediction (Kaggle)](https://www.kaggle.com/c/zillow-prize-1)
- [Zillow Research Data](https://www.zillow.com/research/data/)

### Running

Open and run the notebook top-to-bottom:

```bash
jupyter notebook home_value_estimation.ipynb
```

Or in Google Colab — update the data path in Cell 2:
```python
df = pd.read_csv('Zillow.csv')   # local
# df = pd.read_csv('/content/Zillow.csv')  # Colab
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📚 References

- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD. https://arxiv.org/abs/1603.02754
- Zillow Research & Zestimate Methodology: https://www.zillow.com/z/zestimate/
- Federal Reserve Home Price Index (FRED): https://fred.stlouisfed.org/series/CSUSHPISA
