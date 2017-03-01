from sklearn.ensemble import RandomForestClassifier

param_grid = {'max_depth': [1, 3, 5, 7, 10], 'max_features': [5, 8, 10, 20]}

grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("best parameters: %s" % grid.best_params_)
print("Training set accuracy: %s" % grid.score(X_train, y_train))
print("Test set accuracy: %s" % grid.score(X_test, y_test))

scores = [x.mean_validation_score for x in grid.grid_scores_]
scores = np.array(scores).reshape(5, 4)
plt.matshow(scores)
plt.xlabel("max_features")
plt.ylabel("max_depth")
