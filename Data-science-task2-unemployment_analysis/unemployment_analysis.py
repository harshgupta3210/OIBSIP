import matplotlib
matplotlib.use('Agg')
# # Task 2 · Unemployment Analysis with Python
# **Objective:** exploratory data analysis of unemployment data to uncover regional and temporal trends, with a focus on the impact of the COVID-19 pandemic on unemployment in India.
# 
# **Dataset:** Kaggle → *Unemployment in India* (`Unemployment in India.csv` or `Unemployment_Rate_upto_11_2020.csv`). Put the CSV in the `data/` folder (or next to this notebook).
# 
# Every chart is also saved as a PNG in the `output/` folder.

# ## 1. Imports and settings

import os, glob
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (11, 5)
os.makedirs('output', exist_ok=True)

# Lockdown in India began on 25 March 2020, so data from March 2020 onwards is treated as 'post-COVID'
COVID_START = pd.Timestamp('2020-03-01')

# ## 2. Load the data, inspect shape, check nulls, convert types

# find the CSV automatically (looks in data/ and in the current folder)
files = glob.glob('data/*.csv') + glob.glob('*.csv')
files = [f for f in files if 'nemploy' in f.lower()] or files
DATA_FILE = files[0]
print('Using file:', DATA_FILE)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()          # the Kaggle file has leading spaces in column names

# give the columns short, clean names
rename = {}
for c in df.columns:
    l = c.lower()
    if l == 'region':            rename[c] = 'State'
    elif l == 'region.1':        rename[c] = 'Zone'
    elif l == 'date':            rename[c] = 'Date'
    elif l == 'frequency':       rename[c] = 'Frequency'
    elif 'unemployment' in l:    rename[c] = 'Unemployment_Rate'
    elif 'employed' in l:        rename[c] = 'Employed'
    elif 'participation' in l:   rename[c] = 'Labour_Participation'
df = df.rename(columns=rename)

print('Shape:', df.shape)
print('\nColumns:', list(df.columns))
print('\nNull values:\n', df.isnull().sum())
print('\nData types (before conversion):\n', df.dtypes)

# remove empty rows, strip text columns, convert Date to datetime
df = df.dropna(how='all').dropna(subset=['State', 'Date', 'Unemployment_Rate'])
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['Period'] = np.where(df['Date'] >= COVID_START, 'Post-COVID', 'Pre-COVID')

print('Data types (after conversion):\n', df.dtypes)
print('\nDate range:', df['Date'].min().date(), 'to', df['Date'].max().date())
print('Number of states/regions:', df['State'].nunique())
print(df.head())
print('\nDescriptive statistics:\n', df.describe().round(2))

# ## 3. EDA - region-wise average unemployment rate

state_avg = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False).round(2)
print(state_avg)

if 'Zone' in df.columns:
    zone_avg = df.groupby('Zone')['Unemployment_Rate'].mean().sort_values(ascending=False)
    zone_avg.plot(kind='bar', color='steelblue')
    plt.title('Average unemployment rate by zone'); plt.ylabel('Unemployment rate (%)'); plt.xticks(rotation=0)
    plt.savefig('output/avg_unemployment_by_zone.png', dpi=110, bbox_inches='tight')
    plt.close("all")

# **Observations:** compare the zones in the chart above. Check which zone has the highest and which the lowest average rate, and whether the gap between them is large. *(Write your own observation here after running.)*

# ## 4. EDA - month-wise trend (all India)

monthly = df.groupby('Date')['Unemployment_Rate'].mean()
plt.plot(monthly.index, monthly.values, marker='o', color='darkred')
plt.axvline(COVID_START, color='gray', linestyle='--', label='COVID-19 lockdown (Mar 2020)')
plt.title('Average unemployment rate over time (all states)')
plt.xlabel('Date'); plt.ylabel('Unemployment rate (%)'); plt.legend()
plt.savefig('output/monthly_trend.png', dpi=110, bbox_inches='tight')
plt.close("all")
peak = monthly.idxmax()
print(f'Highest national average: {monthly.max():.2f}% in {peak.strftime("%B %Y")}')
print(f'Lowest national average: {monthly.min():.2f}% in {monthly.idxmin().strftime("%B %Y")}')

# **Observations:** look at what happens to the line around the dashed lockdown marker, and how quickly it comes back down afterwards. *(Write your own observation here after running.)*

# ## 5. Time-series line chart for 3 or more states

preferred = ['Maharashtra', 'Delhi', 'Tamil Nadu', 'Uttar Pradesh', 'Kerala']
states_to_plot = [s for s in preferred if s in df['State'].unique()][:4]
if len(states_to_plot) < 3:                      # fall back to the 3 states with most data
    states_to_plot = df['State'].value_counts().index[:3].tolist()

plt.figure(figsize=(12, 5))
for s in states_to_plot:
    sub = df[df['State'] == s].sort_values('Date')
    plt.plot(sub['Date'], sub['Unemployment_Rate'], marker='o', label=s)
plt.axvline(COVID_START, color='gray', linestyle='--', label='Lockdown')
plt.title('Unemployment rate over time - selected states')
plt.xlabel('Date'); plt.ylabel('Unemployment rate (%)'); plt.legend()
plt.savefig('output/state_time_series.png', dpi=110, bbox_inches='tight')
plt.close("all")

# **Observations:** do all the states react to the lockdown in the same way, or do some rise much more than others? *(Write your own observation here after running.)*

# ## 6. Bar chart - top 10 states by average unemployment rate

top10 = state_avg.head(10)
sns.barplot(x=top10.values, y=top10.index, hue=top10.index, palette='Reds_r', legend=False)
plt.title('Top 10 states by average unemployment rate'); plt.xlabel('Average unemployment rate (%)')
for i, v in enumerate(top10.values):
    plt.text(v + 0.2, i, f'{v:.1f}%', va='center')
plt.savefig('output/top10_states.png', dpi=110, bbox_inches='tight')
plt.close("all")

# **Observations:** note which states are at the top and how far they are from the next ones. *(Write your own observation here after running.)*

# ## 7. Heatmap - correlation between unemployment, employment and labour participation
# The dataset gives the *number of employed people* (`Employed`) rather than an employment rate, so that column is used as the employment measure.

cols = [c for c in ['Unemployment_Rate', 'Employed', 'Labour_Participation'] if c in df.columns]
corr = df[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Correlation heatmap')
plt.savefig('output/correlation_heatmap.png', dpi=110, bbox_inches='tight')
plt.close("all")
print(corr.round(2))

# **Observations:** check the sign and strength of the relationship between unemployment rate and labour participation. *(Write your own observation here after running.)*

# ## 8. Pre-COVID vs post-COVID comparison

print('Rows per period:\n', df['Period'].value_counts())
period_mean = df.groupby('Period')['Unemployment_Rate'].mean().reindex(['Pre-COVID', 'Post-COVID'])
print('\nMean unemployment rate:\n', period_mean.round(2))

change = df.pivot_table(index='State', columns='Period', values='Unemployment_Rate', aggfunc='mean')
change['Change'] = change['Post-COVID'] - change['Pre-COVID']
change = change.sort_values('Change', ascending=False)
print('\nBiggest increases:\n', change.head(5).round(2))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
period_mean.plot(kind='bar', color=['seagreen', 'crimson'], ax=ax[0])
ax[0].set_title('Mean unemployment rate: pre vs post COVID'); ax[0].set_ylabel('%'); ax[0].tick_params(axis='x', rotation=0)
top_change = change['Change'].dropna().head(10)
sns.barplot(x=top_change.values, y=top_change.index, hue=top_change.index, palette='Reds_r', legend=False, ax=ax[1])
ax[1].set_title('Top 10 states: increase in unemployment rate'); ax[1].set_xlabel('Percentage-point change')
plt.tight_layout()
plt.savefig('output/pre_post_covid.png', dpi=110, bbox_inches='tight')
plt.close("all")

# **Observations:** compare the two means and see which states were hit hardest. *(Write your own observation here after running.)*

# ## 9. Summary of results

pre, post = period_mean['Pre-COVID'], period_mean['Post-COVID']
print(f'Mean unemployment rate before COVID : {pre:.2f}%')
print(f'Mean unemployment rate after COVID  : {post:.2f}%')
print(f'Change                              : {post - pre:+.2f} percentage points')
print(f'State with highest average rate     : {state_avg.index[0]} ({state_avg.iloc[0]:.2f}%)')
print(f'State with lowest average rate      : {state_avg.index[-1]} ({state_avg.iloc[-1]:.2f}%)')
print(f'State with biggest rise after COVID : {change.index[0]} ({change["Change"].iloc[0]:+.2f} points)')
print(f'Peak national average               : {monthly.max():.2f}% in {monthly.idxmax().strftime("%B %Y")}')

# ## 10. Conclusion
# *(Write 3-4 lines using the numbers printed in the summary above: how much unemployment rose after COVID, which states were affected most, and what the correlation heatmap showed.)*
