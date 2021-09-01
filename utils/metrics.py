from sklearn.metrics import precision_score, recall_score, f1_score, plot_precision_recall_curve
import matplotlib.pyplot as plt


def precision_recall_f1(y_true, y_pred):
    _precision = precision_score(y_true, y_pred)
    _recall = recall_score(y_true, y_pred)
    _f1 = f1_score(y_true, y_pred)
    return _precision, _recall, _f1


def precision_recall_curve(model, x_test, y_test):
    pr_plot = plot_precision_recall_curve(model, x_test, y_test)
    return pr_plot.ax_.set_title('Precision-Recall curve')
