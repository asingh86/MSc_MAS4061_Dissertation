from transformer import Transformer
from model import logistic_regression
from utils import metrics


def logistic_regression_results():
    t = Transformer()
    log_reg = logistic_regression
    metric = metrics
    x_train, x_test, y_train, y_test = t.build_features()

    model = log_reg.fit_logistic_regression(x_train, y_train)
    y_pred = log_reg.predict_logistic_regression(model, x_test)

    precision, recall, f1_score = metric.precision_recall_f1(y_test, y_pred)

    return print(f'Precision: {precision}, Recall: {recall} and f1_score: {f1_score}')
