# Task 1 · Iris Flower Classification

Train a machine learning model that identifies the species of an iris flower (Setosa, Versicolor or Virginica) from its physical measurements.

**Tech stack:** Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook
**Dataset:** built into scikit-learn (`load_iris()`), no download needed.

## Folder contents

```
task1_iris_classification/
├── iris_classification.ipynb   # full notebook with code, comments and saved outputs
├── iris_classification.py      # same code as a plain Python script
├── README.md
└── output/
    ├── results.txt             # printed results (EDA, reports, comparison)
    ├── pairplot.png
    ├── boxplots.png
    ├── correlation_heatmap.png
    ├── confusion_matrices.png
    └── model_comparison.png
```

## What was done

- EDA: shape, dtypes, null check, descriptive statistics
- Visualisations: pairplot and box plots by species
- Feature selection discussion: petal length and petal width are the most discriminative features
- 80/20 train/test split (stratified, `random_state=42`)
- Four classifiers: Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest
- Evaluation: accuracy, confusion matrix, classification report (precision, recall, F1), plus 5-fold cross-validation
- Best model declared with justification

## Results

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| Logistic Regression | 0.9667 | 0.9733 |
| K-Nearest Neighbours | 1.0000 | 0.9733 |
| Decision Tree | 0.9333 | 0.9533 |
| Random Forest | 0.9000 | 0.9667 |

**Best model: K-Nearest Neighbours.** It ties with Logistic Regression on cross-validated accuracy and has the higher test accuracy. The test set has only 30 flowers, so the difference between the top models is small. Setosa is classified correctly by every model; the few errors are between Versicolor and Virginica.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook iris_classification.ipynb
```

Or run the script: `python iris_classification.py` (create an `output` folder first if it does not exist).
