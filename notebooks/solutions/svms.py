print("default score without scaling: %f" % SVC().fit(X_train, y_train).score(X_test, y_test))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("default score with scaling: %f" % SVC().fit(X_train_scaled, y_train).score(X_test_scaled, y_test))

grid_search.fit(X_train_scaled, y_train)

# We extract just the scores
scores = [x[1] for x in grid_search.grid_scores_]
scores = np.array(scores).reshape(6, 6)

plt.matshow(scores)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(6), param_grid['gamma'])
plt.yticks(np.arange(6), param_grid['C'])
