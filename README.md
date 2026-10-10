# Data Science / Machine Learning Tasks

A collection of three beginner-to-intermediate projects built with Python and Jupyter Notebook.

| # | Task | Type | Notebook |
|---|------|------|----------|
| 1 | Iris Flower Classification | Classification | `iris_classification.ipynb` |
| 2 | Unemployment Analysis with Python | EDA / Time series | `unemployment_analysis.ipynb` |
| 3 | Car Price Prediction with Machine Learning | Regression | `car_price_prediction.ipynb` |

**Tech stack (all tasks):** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

---

## Task 1 · Iris Flower Classification

**Objective:** Train a classification model that identifies the species of an iris flower (Setosa, Versicolor or Virginica) from its physical measurements.

**Dataset:** Built into scikit-learn (`sklearn.datasets.load_iris()`), no download needed.

**What's done:**
- Load the dataset and run EDA: shape, dtypes, null check, descriptive statistics
- Visualisations: pairplot and box plots for each feature, split by species
- Feature selection discussion: which features discriminate best
- 80/20 train/test split using `train_test_split`
- Train at least two classifiers (e.g. Logistic Regression, KNN, Decision Tree, Random Forest)
- Evaluate with accuracy, confusion matrix and classification report (precision, recall, F1)
- Declare the best model with justification

**Result:** _best model: ____ , accuracy: _____

---

## Task 2 · Unemployment Analysis with Python

**Objective:** Exploratory data analysis of unemployment data to uncover regional and temporal trends, focusing on the impact of COVID-19 on unemployment in India.

**Dataset:** "Unemployment in India" from [Kaggle](https://www.kaggle.com/) (search "unemployment rate India dataset").

**What's done:**
- Data loading, shape inspection, null check and type conversion (dates)
- Region-wise average unemployment rates and month-wise trends
- Time-series line chart for at least 3 major states/regions
- Bar chart of the top 10 states by average unemployment rate
- Heatmap: correlation between unemployment rate, employment rate and labour participation rate
- Pre-COVID vs post-COVID comparison (split by date, compare mean rates)
- Markdown observations between each chart

**Key findings:** _fill in after running the notebook_

---

## Task 3 · Car Price Prediction with Machine Learning

**Objective:** Build a regression model that predicts the selling price of a used car from features such as brand, age, mileage (km driven), fuel type and transmission.

**Dataset:** "Vehicle dataset from CarDekho" from [Kaggle](https://www.kaggle.com/) (`CAR DETAILS FROM CAR DEKHO.csv`).

**What's done:**
- Data cleaning: nulls, duplicates, inconsistent categories (e.g. "Petrol" vs "petrol")
- Feature engineering: `car_age` from the year column, `brand` from the car name
- EDA: price distribution, price vs fuel type (box plot), price vs car age (scatter plot)
- One-hot encoding of categorical variables
- Feature correlation heatmap
- Train/test split
- Models: Linear Regression, Random Forest, Gradient Boosting
- Evaluation with MAE, RMSE and R²
- Feature importance chart for the best model

**Results:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | - | - | - |
| Random Forest | - | - | - |
| Gradient Boosting | - | - | - |

**Best model:** _fill in after running the notebook_

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Open the notebook you want and run all cells. For Tasks 2 and 3, download the dataset from Kaggle first and keep the CSV in the same folder as the notebook.

## Project Structure

```
.
├── README.md
├── iris_classification.ipynb
├── unemployment_analysis.ipynb
├── car_price_prediction.ipynb
└── data/            (Kaggle CSV files for Tasks 2 and 3)
```
