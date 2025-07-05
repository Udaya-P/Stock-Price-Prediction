# 📈 Stock Price Prediction using Machine Learning

Welcome to the Stock Price Prediction project! This repository demonstrates how to use machine learning models to predict the future trend (up or down) of a stock's closing price based on historical data. The project uses Tesla stock data and includes data preprocessing, feature engineering, model training, and evaluation.

---

## 📚 Project Overview

This project aims to:
- Predict whether the stock price will go **up or down** the next day.
- Use **historical features** like Open, High, Low, Close, Adj Close, and Volume.
- Apply **classification algorithms**: Logistic Regression, Support Vector Machine, and XGBoost.
- Evaluate using **ROC-AUC Score** for better probabilistic insight.
- Visualize insights using Seaborn and Matplotlib.

---

## 🧠 Machine Learning Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| SVM (Polynomial Kernel) | Non-linear model to capture complex patterns |
| XGBoost Classifier | Powerful ensemble tree-based method |

All models are evaluated using:
- **ROC-AUC Score** (on training and validation sets)
- **Soft probabilities** (`predict_proba`) for better decision-making insights

---

## 📊 Dataset

- **Source**: https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021
- **Features**:
  - `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
  - Engineered: `day`, `month`, `year`, `target` (next-day trend)

---

## 🧪 Evaluation Metric

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** is used because:
- It evaluates **ranking quality**, not just classification accuracy
- It works well with **imbalanced classes**
- It allows us to compare **soft predictions** between 0 and 1

---
## 📈 Visualizations Included

- Boxplots to detect outliers
- Feature distributions
- ROC-AUC scores printed per model
- Optionally: correlation heatmap

---

## ✅ Key Findings

- Logistic Regression performs consistently well with stable ROC-AUC on both train and validation sets.
- SVM may overfit if not tuned properly.
- XGBoost performs the best but needs regularization to avoid overfitting.
- Feature engineering (like extracting day/month/year) improves model accuracy.

---

## 🚀 Future Work

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Use of LSTM or RNN for time-series prediction
- Incorporate technical indicators (RSI, MACD, Moving Averages)
- Deploy as a web app using Streamlit or FastAPI

---

## 🧑‍💻 Author

- **Udaya Pragna Gangavaram**
- GitHub: [@Udaya-P](https://github.com/Udaya-P)

---

## 📜 License

This project is open-source and free to use for educational and research purposes.

```
