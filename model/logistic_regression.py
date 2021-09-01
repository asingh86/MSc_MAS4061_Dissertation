from sklearn.linear_model import LogisticRegression


def fit_logistic_regression(x_train, y_train):
    model = LogisticRegression()
    model = model.fit(x_train, y_train)
    return model


def predict_logistic_regression(model, x_test):
    return model.predict(x_test)
