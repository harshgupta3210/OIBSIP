# Data Science & Machine Learning Tasks (Python)

This repository contains three separate, runnable Python projects built in Jupyter Notebook:

1. `task1_iris_classification` - Iris Flower Classification
2. `task2_unemployment_analysis` - Unemployment Analysis with Python
3. `task3_car_price_prediction` - Car Price Prediction with Machine Learning

**Tech stack (all projects):** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

| # | Project | Type | Dataset |
|---|---------|------|---------|
| 1 | Iris Flower Classification | Classification | Built into scikit-learn (no download) |
| 2 | Unemployment Analysis | EDA / time series | Kaggle: "Unemployment in India" |
| 3 | Car Price Prediction | Regression | Kaggle: "Vehicle dataset from CarDekho" |

## Requirements
- Python 3.10+ recommended
- Windows instructions below; macOS/Linux are similar
- Install dependencies inside a virtual environment

## Run any project (Windows PowerShell)
```powershell
cd task1_iris_classification
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Open the `.ipynb` file and choose **Run All**. Repeat for each folder. On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.

---

## Task 1 · Iris Flower Classification

**Objective:** Train a classification model that identifies the species of an iris flower (Setosa, Versicolor or Virginica) from its physical measurements.

**Dataset:** Built into scikit-learn (`load_iris()`), no download needed.

**What it covers:**
- EDA: shape, dtypes, null check, descriptive statistics
- Pairplot and box plots by species
- Feature selection discussion (petal length and width are the most discriminative)
- 80/20 stratified train/test split
- Four classifiers: Logistic Regression, KNN, Decision Tree, Random Forest
- Accuracy, confusion matrix, classification report, 5-fold cross-validation
- Best model with justification

**Results:**

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| Logistic Regression | 0.9667 | 0.9733 |
| K-Nearest Neighbours | 1.0000 | 0.9733 |
| Decision Tree | 0.9333 | 0.9533 |
| Random Forest | 0.9000 | 0.9667 |

**Best model:** K-Nearest Neighbours (ties Logistic Regression on CV accuracy, higher test accuracy). The test set has only 30 flowers, so the gap between top models is small.

---

## Task 2 · Unemployment Analysis with Python

**Objective:** Exploratory data analysis of unemployment data to uncover regional and temporal trends, focusing on the impact of COVID-19 on unemployment in India.

**Dataset:** "Unemployment in India" from Kaggle (free account needed). Place the CSV in the project folder; if file or column names differ, update the load cell.

**What it covers:**
- Data loading, shape inspection, null check, type conversion (dates)
- Region-wise average unemployment rates and month-wise trends
- Time-series line chart for at least 3 major states/regions
- Bar chart: top 10 states by average unemployment rate
- Heatmap: correlation between unemployment rate, employment and labour participation rate
- Pre-COVID vs post-COVID comparison (mean rates for each period)
- Markdown observations between each chart

**Key findings:** _fill in after running the notebook._

---

## Task 3 · Car Price Prediction with Machine Learning

**Objective:** Build a regression model that predicts the selling price of a used car from brand, age, mileage (km driven), fuel type and transmission.

**Dataset:** "Vehicle dataset from CarDekho" from Kaggle (`CAR DETAILS FROM CAR DEKHO.csv`). Place it in the project folder; if the file name differs, change it in the load cell.

**What it covers:**
- Data cleaning: nulls, duplicates, inconsistent categories (e.g. "Petrol" vs "petrol")
- Feature engineering: `car_age` from the year column, `brand` from the car name
- EDA: price distribution, price vs fuel type (box plot), price vs car age (scatter plot)
- One-hot encoding, correlation heatmap, train/test split
- Three models: Linear Regression, Random Forest, Gradient Boosting
- Evaluation with MAE, RMSE and R², and a feature importance chart for the best model

**Results:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | - | - | - |
| Random Forest | - | - | - |
| Gradient Boosting | - | - | - |

**Best model:** _fill in after running the notebook._

---

## Project structure
```
.
├── README.md
├── task1_iris_classification/
├── task2_unemployment_analysis/
└── task3_car_price_prediction/
```

## GitHub upload
```bash
git init
git add .
git commit -m "Add Iris, Unemployment Analysis and Car Price Prediction projects"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Do not upload `.venv`. Very large dataset files can be left out; link the Kaggle page instead.

## Important notes
- Dataset downloads need internet access and a free Kaggle account.
- Run each notebook with **Restart & Run All** before uploading so outputs are saved inside it.
- Results can vary slightly with a different dataset version or random seed.
- Only report results you actually obtained from your own run.
