# Task 3 · Car Price Prediction with Machine Learning

A regression project that predicts the selling price of a used car from its brand, age, mileage (km driven), fuel type and transmission.

**Tech stack:** Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook
**Dataset:** "Vehicle dataset from CarDekho" from Kaggle (`CAR DETAILS FROM CAR DEKHO.csv`). Search "car price prediction dataset" on Kaggle.

## Folder contents

```
task3_car_price_prediction/
├── car_price_prediction.ipynb   # full notebook with code and comments
├── car_price_prediction.py      # same code as a plain Python script
├── README.md
├── data/                        # put the Kaggle CSV here
└── output/                      # charts and results are saved here when you run the notebook
```

After you run the notebook, `output/` will contain: `price_distribution.png`, `price_vs_fuel_boxplot.png`, `price_vs_age_scatter.png`, `correlation_heatmap.png`, `actual_vs_predicted.png`, `feature_importance.png` and `model_results.csv`.

## What was done

1. Data cleaning: null values, duplicates, inconsistent categories (e.g. "Petrol" vs "petrol")
2. Feature engineering: `car_age` from the year column, `brand` extracted from the car name
3. EDA: price distribution, price vs fuel type (box plot), price vs car age (scatter plot)
4. One-hot encoding of categorical variables
5. Feature correlation heatmap
6. Train/test split (80/20)
7. Three regression models: Linear Regression, Random Forest, Gradient Boosting
8. Evaluation with MAE, RMSE and R² (in rupees; models are trained on log price because prices are skewed)
9. Feature importance chart for the best model
10. Markdown observation cell after each chart and a printed summary of results

## How to run (Windows PowerShell)

```powershell
cd task3_car_price_prediction
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook car_price_prediction.ipynb
```

1. Download the CSV from Kaggle and put it in the `data/` folder.
2. Open the notebook and choose **Kernel → Restart & Run All**.
3. The charts and printed results are saved inside the notebook and in `output/`.

## Results

_Fill in after running the notebook with your dataset (the last code cell prints these numbers and saves them to `output/model_results.csv`):_

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | - | - | - |
| Random Forest | - | - | - |
| Gradient Boosting | - | - | - |

**Best model:** _fill in after running the notebook._

## Notes

- The CarDekho file has `km_driven` instead of a mileage column, so it is used as mileage.
- Brands with fewer than 20 cars are grouped into "Other".
- I could not download the dataset where this notebook was built, so it was tested only on dummy data to check that the code runs. All real numbers come from your own run.
- Write your own sentences in the observation cells and the conclusion after you see your charts.
- Do not upload `.venv`. Very large CSV files can be left out; link the Kaggle page instead.
