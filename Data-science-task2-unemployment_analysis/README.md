# Task 2 · Unemployment Analysis with Python

Exploratory data analysis of unemployment data to uncover regional and temporal trends, with a focus on the impact of the COVID-19 pandemic on unemployment in India.

**Tech stack:** Python, pandas, matplotlib, seaborn, Jupyter Notebook
**Dataset:** "Unemployment in India" from Kaggle (free account needed). Search "unemployment rate India dataset".

## Folder contents

```
task2_unemployment_analysis/
├── unemployment_analysis.ipynb   # full notebook with code and comments
├── unemployment_analysis.py      # same code as a plain Python script
├── README.md
├── data/                         # put the Kaggle CSV here
└── output/                       # charts are saved here when you run the notebook
```

After you run the notebook, `output/` will contain: `avg_unemployment_by_zone.png`, `monthly_trend.png`, `state_time_series.png`, `top10_states.png`, `correlation_heatmap.png`, `pre_post_covid.png`.

## What was done

- Data loading, shape inspection, null check and type conversion (dates, clean column names)
- Region-wise average unemployment rates (by state and by zone)
- Month-wise trend of the all-India average, with the lockdown marked
- Time-series line chart for 3+ major states
- Bar chart: top 10 states with the highest average unemployment rate
- Heatmap: correlation between unemployment rate, employment and labour participation rate
- Pre-COVID vs post-COVID comparison (split at March 2020, mean rates for each period, states with the biggest rise)
- Markdown observation cell after each chart and a printed summary of results

## How to run (Windows PowerShell)

```powershell
cd task2_unemployment_analysis
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook unemployment_analysis.ipynb
```

1. Download the CSV from Kaggle and put it in the `data/` folder.
2. Open the notebook and choose **Kernel → Restart & Run All**.
3. The charts and printed results are saved inside the notebook and in `output/`.

## Results

_Fill in after running the notebook with your dataset (the last code cell prints all the numbers):_

| Measure | Value |
|---|---|
| Mean unemployment rate before COVID | - |
| Mean unemployment rate after COVID | - |
| State with highest average rate | - |
| State with biggest rise after COVID | - |
| Peak national average (month) | - |

## Notes

- The notebook finds the CSV automatically in `data/` or in the notebook folder. If it fails to read your file, check the column names; the code expects the standard Kaggle ones (Region, Date, Frequency, Estimated Unemployment Rate, Estimated Employed, Estimated Labour Participation Rate).
- The dataset has the number of employed people, not an employment rate, so that column is used as the employment measure in the heatmap.
- I could not download the dataset where this notebook was built, so it was tested only on dummy data to check that the code runs. All real numbers come from your own run.
- Write your own sentences in the observation cells and the conclusion after you see your charts.
- Do not upload `.venv`. Very large CSV files can be left out; link the Kaggle page instead.
