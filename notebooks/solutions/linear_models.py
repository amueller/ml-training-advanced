from pprint import pprint

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target % 2)

grid = GridSearchCV(LinearSVC(), param_grid={'C': np.logspace(-6, 2, 9)}, cv=5)
grid.fit(X_train, y_train)
pprint(grid.grid_scores_)
pprint(grid.score(X_test, y_test))


Cs = [10, 1, .01, 0.001, 0.0001]
for penalty in ['l1', 'l2']:
    svm_models = {}
    training_scores = []
    test_scores = []
    for C in Cs:
        svm = LinearSVC(C=C, penalty=penalty, dual=False).fit(X_train, y_train)
        training_scores.append(svm.score(X_train, y_train))
        test_scores.append(svm.score(X_test, y_test))
        svm_models[C] = svm

    plt.figure()
    plt.plot(training_scores, label="training scores")
    plt.plot(test_scores, label="test scores")
    plt.xticks(range(4), Cs)
    plt.legend(loc="best")

    plt.figure(figsize=(10, 5))
    for i, C in enumerate(Cs):
        plt.plot(svm_models[C].coef_.ravel(), "o", label="C = %.2f" % C)

    plt.legend(loc="best")
