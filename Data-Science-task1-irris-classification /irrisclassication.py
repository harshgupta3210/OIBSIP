import matplotlib
matplotlib.use('Agg')
# # Task 1 · Iris Flower Classification
# **Objective:** train a classification model that identifies the species of an iris flower (Setosa, Versicolor or Virginica) from its physical measurements.
# 
# **Dataset:** built into scikit-learn (`load_iris()`), no download needed.

# ## 1. Imports and loading the dataset

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sns.set_style('whitegrid')

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))
print(df.head())

# ## 2. Exploratory Data Analysis

print('Shape:', df.shape)
print('\nData types:\n', df.dtypes)
print('\nNull values:\n', df.isnull().sum())
print('\nClass counts:\n', df['species'].value_counts())
print('\nDescriptive statistics:\n', df.drop(columns='target').describe().round(2))

# ## 3. Visualisations

# **Pairplot** - how each pair of features separates the three species.

g = sns.pairplot(df.drop(columns='target'), hue='species', palette='Set1')
g.figure.suptitle('Iris pairplot by species', y=1.02)
plt.savefig('output/figure.png'); plt.close('all')

# **Box plots** - the spread of each feature per species.

features = iris.feature_names
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(x='species', y=col, data=df, hue='species', palette='Set2', legend=False, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig('output/figure.png'); plt.close('all')

# ## 4. Feature selection discussion

plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature correlation')
plt.savefig('output/figure.png'); plt.close('all')
print('Mean of each feature per species:')
print(df.groupby('species')[features].mean().round(2))

# **Which features discriminate best?** Petal length and petal width separate the three species far better than sepal length and sepal width. Setosa is completely separable using the petal measurements alone, while Versicolor and Virginica overlap slightly. The sepal width is the weakest feature because the species overlap heavily on it. We still keep all four features, since the dataset is small and the extra features can only help.

# ## 5. Train / test split (80 / 20)

X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)   # stratify keeps class balance equal
print('Train:', X_train.shape, ' Test:', X_test.shape)

# ## 6. Train the classifiers

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print('Trained:', name)

# ## 7. Evaluate each model
# Accuracy, confusion matrix and classification report (precision, recall, F1).

results = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, model) in zip(axes.ravel(), models.items()):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(model, X, y, cv=5).mean()
    results[name] = {'test_accuracy': acc, 'cv_accuracy': cv}
    print('=' * 60)
    print(name)
    print('Test accuracy:', round(acc, 4), '| 5-fold CV accuracy:', round(cv, 4))
    print(classification_report(y_test, pred, target_names=iris.target_names))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_title(f'{name} (acc={acc:.2f})'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('output/figure.png'); plt.close('all')

# ## 8. Compare models and pick the best one

summary = pd.DataFrame(results).T.round(4)
print(summary)
best = summary.sort_values(['cv_accuracy', 'test_accuracy'], ascending=False).index[0]
print('\nBest model:', best)
summary.plot(kind='bar', figsize=(9, 5), ylim=(0.8, 1.02))
plt.title('Model comparison'); plt.ylabel('Accuracy'); plt.xticks(rotation=20)
plt.savefig('output/figure.png'); plt.close('all')

# ## 9. Conclusion
# **Best model: K-Nearest Neighbours**
# 
# - Test accuracy: 1.0000, 5-fold cross-validated accuracy: 0.9733.
# - The model was chosen on cross-validated accuracy, with test accuracy as tie-breaker. The test set has only 30 flowers, so one mistake changes accuracy by 3%, and cross-validation gives a more reliable ranking.
# - Almost all errors involve Versicolor vs Virginica; Setosa is always classified correctly.
# - Petal length and petal width are the most useful features.
