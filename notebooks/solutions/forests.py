from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.learning_curve import validation_curve

digits = load_digits()

def plot_validation_curve(parameter_values, train_scores, validation_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.fill_between(parameter_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(parameter_values, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.plot(parameter_values, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(parameter_values, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.ylim(validation_scores_mean.min() - .1, train_scores_mean.max() + .1)
    plt.legend(loc="best")

param_range = range(1, 50)
training_scores, validation_scores = validation_curve(DecisionTreeClassifier(), digits.data, digits.target,
                                                      param_name="max_depth",
                                                      param_range=param_range,
                                                      cv=5)
plt.figure()
plot_validation_curve(param_range, training_scores, validation_scores)

param_range = range(1, 20, 1)
training_scores, validation_scores = validation_curve(RandomForestClassifier(n_estimators=100),
                                                      digits.data, digits.target,
                                                      param_name="max_features",
                                                      param_range=param_range,
                                                      cv=5)
plt.figure()
plot_validation_curve(param_range, training_scores, validation_scores)
