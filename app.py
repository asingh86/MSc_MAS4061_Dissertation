from transformer import Transformer
from model import logistic_regression, lda
from utils import metrics, extractor, text_processing
from gensim import corpora
import pandas


def lda_hyperparameter_extraction() -> pd.DataFrame:
    e = extractor.DataExtractor()
    tp = text_processing.TextProcessing()

    train_reviews, train_labels, test_reviews, test_labels = e.process_freq_text()
    clean_list, id2word, corpus = tp.lda_model_data(train_reviews)

    ld = lda.LDA(corpus=corpus, id2word=id2word, clean_list=clean_list)
    lda_cv_df = ld.lda_cross_validation()
    return lda_cv_df

# todo: this function need updating
def lda_model_train_vis_predict():
    t = Transformer()
    e = extractor.DataExtractor()
    tp = text_processing.TextProcessing()

    train_reviews, train_labels, test_reviews, test_labels = e.process_freq_text()
    clean_list, id2word, corpus = tp.lda_model_data(train_reviews)
    lda_model = lda.build_lda_model(corpus=corpus, id2word=id2word)
    coherence_score = lda.coherence_score(lda_model, id2word, clean_list)

    test_topics = []
    clean_reviews = tp.lda_test_data_processing(test_reviews)
    for clean_review in clean_reviews:
        if clean_review:
            test_topic = lda_model[id2word.doc2bow(clean_review)]
        else:
            test_topic = -1
        test_topics.append(test_topic)

    return lda_model, coherence_score, test_topics


def logistic_regression_results():
    t = Transformer()
    log_reg = logistic_regression
    metric = metrics
    x_train, x_test, y_train, y_test = t.build_features()

    model = log_reg.fit_logistic_regression(x_train, y_train)
    y_pred = log_reg.predict_logistic_regression(model, x_test)

    precision, recall, f1_score = metric.precision_recall_f1(y_test, y_pred)

    return print(f'Precision: {precision}, Recall: {recall} and f1_score: {f1_score}')

