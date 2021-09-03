from transformer import Transformer
from model import logistic_regression, lda
from utils import metrics, extractor, text_processing
from gensim import corpora


def logistic_regression_results():
    t = Transformer()
    log_reg = logistic_regression
    metric = metrics
    x_train, x_test, y_train, y_test = t.build_features()

    model = log_reg.fit_logistic_regression(x_train, y_train)
    y_pred = log_reg.predict_logistic_regression(model, x_test)

    precision, recall, f1_score = metric.precision_recall_f1(y_test, y_pred)

    return print(f'Precision: {precision}, Recall: {recall} and f1_score: {f1_score}')


def lda_model_results():
    t = Transformer()
    e = extractor.DataExtractor()
    tp = text_processing.TextProcessing()

    # get data
    train_reviews, train_labels, test_reviews, test_labels = e.process_freq_text()

    # perform preprocessing
    clean_list = tp.lda_processing(train_reviews)
    id2word = corpora.Dictionary(clean_list)
    texts = clean_list
    corpus = [id2word.doc2bow(text) for text in texts]

    # build model
    lda_model = lda.build_lda_model(corpus=corpus, id2word=id2word)

    test_topics = []
    clean_reviews = tp.lda_test_data_processing(test_reviews)
    for clean_review in clean_reviews:
        if clean_review:
            test_topic = lda_model[id2word.doc2bow(clean_review)]
        else:
            test_topic = -1
        test_topics.append(test_topic)

    return test_topics
