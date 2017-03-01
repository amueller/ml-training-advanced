from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

rng = np.random.RandomState(42)
iris = load_iris()
X = np.hstack([iris.data, rng.uniform(size=(len(iris.data), 5))])
X_train, X_test, y_train, y_test = train_test_split(X, iris.target, random_state=2)

selection_pipe = make_pipeline(SelectKBest(), LinearSVC())
param_grid = {'linearsvc__C': 10. ** np.arange(-3, 3),
              'selectkbest__k': [1, 2, 3, 4, 5, 7]}
grid = GridSearchCV(selection_pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best parameters: %s" % grid.best_params_)
print("Test set performance: %s" % grid.score(X_test, y_test))
