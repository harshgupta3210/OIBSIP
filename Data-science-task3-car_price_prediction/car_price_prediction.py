import matplotlib
matplotlib.use('Agg')
# # Task 3 · Car Price Prediction with Machine Learning
# **Objective:** build a regression model that predicts the selling price of a used car from its brand, age, mileage (km driven), fuel type and transmission.
# 
# **Dataset:** Kaggle → *Vehicle dataset from CarDekho* (`CAR DETAILS FROM CAR DEKHO.csv`). Put the CSV in the `data/` folder (or next to this notebook).
# 
# Every chart is also saved as a PNG in the `output/` folder.

# ## 1. Imports and settings

import os, glob
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sns.set_style('whitegrid')
os.makedirs('output', exist_ok=True)

CURRENT_YEAR = 2026     # used to calculate the age of the car

# ## 2. Load the data

# find the CSV automatically (looks in data/ and in the current folder)
files = glob.glob('data/*.csv') + glob.glob('*.csv')
files = [f for f in files if 'car' in f.lower()] or files
DATA_FILE = files[0]
print('Using file:', DATA_FILE)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
print('Shape:', df.shape)
print(df.head())
print('\nData types:\n', df.dtypes)
print('\nDescriptive statistics:\n', df.describe().round(2))

# ## 3. Data cleaning
# Handle null values, remove duplicates and fix inconsistent categorical values (e.g. `Petrol` vs `petrol`).

print('Null values:\n', df.isnull().sum())
df = df.dropna()

print('\nDuplicate rows:', df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# strip spaces and use one consistent case in every text column
for col in ['fuel', 'seller_type', 'transmission', 'owner']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        print(col, '->', sorted(df[col].unique()))
print('\nShape after cleaning:', df.shape)

# ## 4. Feature engineering
# - **car_age** = current year − manufacturing year
# - **brand** = first word of the car name (`Land Rover` is a special case because its name has two words)

df['car_age'] = CURRENT_YEAR - df['year']

df['brand'] = df['name'].str.strip().str.split().str[0].str.title()
df['brand'] = df['brand'].replace({'Land': 'Land Rover'})
print(df['brand'].value_counts().head(10))

# group very rare brands into 'Other' so one-hot encoding stays small
counts = df['brand'].value_counts()
df['brand'] = df['brand'].where(df['brand'].isin(counts[counts >= 20].index), 'Other')
print('\nBrands used:', df['brand'].nunique())

df = df.drop(columns=['name', 'year'])
print(df.head())

# ## 5. Exploratory Data Analysis

# **Distribution of selling prices** (the log scale on the right is much closer to a bell shape).

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df['selling_price'], bins=50, kde=True, ax=ax[0]); ax[0].set_title('Selling price')
sns.histplot(np.log1p(df['selling_price']), bins=50, kde=True, ax=ax[1]); ax[1].set_title('log(Selling price)')
plt.savefig('output/price_distribution.png', dpi=110, bbox_inches='tight')
plt.close("all")
print('Skewness of price:', round(df['selling_price'].skew(), 2))

# **Observations:** check whether the price distribution is skewed and how many cars sit at the very expensive end. *(Write your own observation here after running.)*

# **Price vs fuel type**

plt.figure(figsize=(9, 5))
sns.boxplot(x='fuel', y='selling_price', data=df, hue='fuel', palette='Set2', legend=False)
plt.yscale('log'); plt.title('Selling price by fuel type (log scale)')
plt.savefig('output/price_vs_fuel_boxplot.png', dpi=110, bbox_inches='tight')
plt.close("all")
print(df.groupby('fuel')['selling_price'].median().sort_values(ascending=False))

# **Observations:** compare the medians of the fuel types. *(Write your own observation here after running.)*

# **Price vs car age**

plt.figure(figsize=(9, 5))
sns.scatterplot(x='car_age', y='selling_price', data=df, alpha=0.4)
plt.yscale('log'); plt.title('Selling price vs car age (log scale)')
plt.savefig('output/price_vs_age_scatter.png', dpi=110, bbox_inches='tight')
plt.close("all")
print('Correlation between car age and price:', round(df['car_age'].corr(df['selling_price']), 2))

# **Observations:** see what happens to the price as the car gets older. *(Write your own observation here after running.)*

# ## 6. Encode categorical variables (One-Hot Encoding)
# The target is converted to log-price because prices are heavily skewed. Predictions are converted back to rupees before evaluation.

y = np.log1p(df['selling_price'])
X = pd.get_dummies(df.drop(columns=['selling_price']), drop_first=True)
print('Feature matrix shape:', X.shape)
print(list(X.columns))

# ## 7. Feature correlation heatmap

corr = X.assign(selling_price=df['selling_price']).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature correlation heatmap')
plt.savefig('output/correlation_heatmap.png', dpi=110, bbox_inches='tight')
plt.close("all")
print(corr['selling_price'].drop('selling_price').sort_values(ascending=False).round(2).head(5))
print(corr['selling_price'].drop('selling_price').sort_values().round(2).head(5))

# ## 8. Train / test split (80 / 20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train:', X_train.shape, ' Test:', X_test.shape)

# ## 9. Train the regression models

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print('Trained:', name)

# ## 10. Evaluate the models (MAE, RMSE, R²)
# Metrics are calculated in the original price unit (rupees).

y_true = np.expm1(y_test)
rows = []
for name, model in models.items():
    pred = np.expm1(model.predict(X_test))
    rows.append({'Model': name,
                 'MAE': mean_absolute_error(y_true, pred),
                 'RMSE': np.sqrt(mean_squared_error(y_true, pred)),
                 'R2': r2_score(y_true, pred)})
results = pd.DataFrame(rows).set_index('Model').round(3)
print(results)

best_name = results['R2'].idxmax()
best_model = models[best_name]
print('\nBest model (highest R2):', best_name)

pred = np.expm1(best_model.predict(X_test))
plt.figure(figsize=(6, 6))
plt.scatter(y_true, pred, alpha=0.4)
lim = [0, max(y_true.max(), pred.max())]
plt.plot(lim, lim, 'r--'); plt.xlabel('Actual price'); plt.ylabel('Predicted price')
plt.title(f'{best_name}: actual vs predicted')
plt.savefig('output/actual_vs_predicted.png', dpi=110, bbox_inches='tight')
plt.close("all")

# **Observations:** compare the three models; a higher R² and lower MAE/RMSE is better. Notice where the predictions drift away from the red line. *(Write your own observation here after running.)*

# ## 11. Feature importance of the best model

if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=X.columns)
else:                                  # linear regression: use the size of the coefficients
    imp = pd.Series(np.abs(best_model.coef_), index=X.columns)
imp = imp.sort_values().tail(15)
imp.plot(kind='barh', figsize=(8, 6), color='teal')
plt.title(f'Top 15 features - {best_name}')
plt.savefig('output/feature_importance.png', dpi=110, bbox_inches='tight')
plt.close("all")
print(imp.sort_values(ascending=False).head(5).round(4))

# ## 12. Summary of results

print(results)
print()
print(f'Best model : {best_name}')
print(f'R2         : {results.loc[best_name, "R2"]:.3f}')
print(f'MAE        : {results.loc[best_name, "MAE"]:,.0f}')
print(f'RMSE       : {results.loc[best_name, "RMSE"]:,.0f}')
results.to_csv('output/model_results.csv')

# ## 13. Conclusion
# *(Write 3-4 lines using the numbers printed above: which model won and its R², MAE and RMSE; which features mattered most; and what you could improve, e.g. hyper-parameter tuning, cross-validation, outlier handling.)*
