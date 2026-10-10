# Data Science & Machine Learning Tasks (Python)

This repository contains three separate, runnable Python projects built in Jupyter Notebook:

1. `task1_iris_classification` - Iris Flower Classification (classification)
2. `task2_unemployment_analysis` - Unemployment Analysis with Python (EDA, India, COVID-19 impact)
3. `task3_car_price_prediction` - Car Price Prediction with Machine Learning (regression)

Each project folder includes its own notebook, README and output (charts/results). Libraries used: pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Project overview

| # | Project | Type | Dataset |
|---|---------|------|---------|
| 1 | Iris Flower Classification | Classification | Built into scikit-learn (no download) |
| 2 | Unemployment Analysis | EDA / time series | Kaggle: "Unemployment in India" |
| 3 | Car Price Prediction | Regression | Kaggle: "Vehicle dataset from CarDekho" |

## Requirements
- Python 3.10+ recommended
- Windows instructions are below; macOS/Linux are similar.
- Install dependencies inside a virtual environment.

## Run any project (Windows PowerShell)
```powershell
cd task1_iris_classification
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Open the `.ipynb` file and choose **Run All**. Repeat for each folder. On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Task-specific setup
- **Iris Classification:** nothing to download. The dataset loads from `sklearn.datasets.load_iris()`.
- **Unemployment Analysis:** download "Unemployment in India" from Kaggle (free account needed) and place the CSV in the project folder. If your file name or column names differ, update the load cell and column names in the notebook.
- **Car Price Prediction:** download "Vehicle dataset from CarDekho" from Kaggle and place `CAR DETAILS FROM CAR DEKHO.csv` in the project folder. If your file name differs, change it in the load cell.

## What each project covers
- **Task 1:** EDA, pairplot and box plots, feature selection discussion, 80/20 split, four classifiers (Logistic Regression, KNN, Decision Tree, Random Forest), accuracy / confusion matrix / classification report, best model with justification.
- **Task 2:** data loading, shape and null checks, type conversion, region-wise and month-wise EDA, line chart for 3+ states, top-10 states bar chart, correlation heatmap, pre-COVID vs post-COVID comparison, markdown observations.
- **Task 3:** data cleaning, feature engineering (car age, brand), EDA, one-hot encoding, correlation heatmap, train/test split, three regression models, MAE / RMSE / R², feature importance chart.

## Results

| Project | Best model / key finding |
|---|---|
| Iris Classification | K-Nearest Neighbours: test accuracy 1.00, 5-fold CV accuracy 0.9733 |
| Unemployment Analysis | _fill in after running the notebook_ |
| Car Price Prediction | _fill in after running the notebook (best model, MAE, RMSE, R²)_ |

## GitHub upload
1. Create a new repository on GitHub (for example `data-science-tasks`).
2. Put the three task folders directly inside the repository root, alongside this README.
3. In Git Bash, from the repository root:
```bash
git init
git add README.md task1_iris_classification task2_unemployment_analysis task3_car_price_prediction
git commit -m "Add Iris, Unemployment Analysis and Car Price Prediction projects"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Do not upload `.venv`. Very large dataset files can be left out; instead, add the Kaggle link in the project README.

## Submission checklist
- Source code (clean, commented notebook) for every project
- A README for every project
- Screenshots of the key outputs and charts (taken from your own run)
- Notebook run with **Restart & Run All** before uploading, so outputs are saved inside it
- Results sections filled in with your own numbers

## Important notes
- Dataset downloads need internet access and a free Kaggle account.
- Results can vary slightly with a different dataset version or random seed.
- Verify every notebook on your own machine before submitting; only report results you actually obtained.
