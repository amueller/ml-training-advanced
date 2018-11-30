import pandas as pd
data = pd.read_csv("data/bank-campaign.csv")
display(data.head())

y = data.target
X = data.drop("target", axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

print("label proportions:", y.value_counts() / len(y))

# Get reasonable tree depth:
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X_train, y_train)
# new in 0.21, use tree.tree_.max_depth in 0.20
print("Tree depth: ", tree.get_depth())
print("n_features: ", X.shape[1])

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'max_depth': [3, 5, 8, 10, 15, 20, 30],
              'max_features':[4, 8, 16, 20, 25, 40]}
grid = GridSearchCV(RandomForestClassifier(n_estimators=100),
                    param_grid=param_grid, cv=5)

# use [::10] to subsample by a factor of 10 for impatience
# could also have used StratifiedShuffleSplit(train_size=.1)
grid.fit(X_train[::10], y_train[::10])

res = pd.DataFrame(grid.cv_results_)
print(res.keys())
res_piv = pd.pivot_table(
    res, values='mean_test_score', index='param_max_depth',
    columns='param_max_features')

display(res_piv)

import matplotlib.pyplot as plt
%matplotlib inline
plt.matshow(res_piv.values)
plt.xlabel(res_piv.columns.name)
plt.xticks(range(res_piv.shape[1]), res_piv.columns)
plt.ylabel(res_piv.index.name)
plt.yticks(range(res_piv.shape[0]), res_piv.index);
plt.colorbar()