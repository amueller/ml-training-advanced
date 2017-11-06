data = pd.read_csv("data/bike_day_raw.csv")
X = data.drop("cnt", axis=1)
y = data.cnt

display(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)

from sklearn.linear_model import LinearRegression


# scale here

lr = LinearRegression().fit(X_train, y_train)

print(lr.score(X_train, y_train))

print(lr.score(X_test, y_test))

from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))

lr = LinearRegression().fit(X_train_ohe, y_train)

print(lr.score(X_train_ohe, y_train))

print(lr.score(X_test_ohe, y_test))

from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_test_ohe)
print(mean_squared_error(y_test, y_pred))