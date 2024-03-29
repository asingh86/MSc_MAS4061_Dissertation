import pandas as pd
from utils.extractor import DataExtractor
from utils.text_processing import TextProcessing
from gensim import corpora


class Transformer:

    def __init__(self):
        self.__extract_data = DataExtractor()
        self.__process_text = TextProcessing()

    def build_freq_feature(self):
        train_reviews, train_labels, test_reviews, test_labels = self.__extract_data.process_freq_text()
        freq_dict = self.__process_text.build_freq(train_reviews, train_labels)

        train_review_feature = self.__process_text.word_count_feature_extraction(train_reviews, train_labels,
                                                                                 freq_dict)
        train_review_feature = pd.DataFrame(train_review_feature, columns=['negative_score', 'positive_score',
                                                                           'target_label'])

        test_review_feature = self.__process_text.word_count_feature_extraction(test_reviews, test_labels,
                                                                                freq_dict)
        test_review_feature = pd.DataFrame(test_review_feature, columns=['negative_score', 'positive_score',
                                                                         'target_label'])
        return train_review_feature, test_review_feature

    def model_data(self):
        files = self.__extract_data.get_data()
        train_review_feature, test_review_feature = self.build_freq_feature()

        selected_columns = ['product_id', 'product_parent', 'product_category',
                            'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
                            'review_headline', 'review_body', 'negative_score',
                            'positive_score', 'target_label']

        train_data = pd.concat([files['train_neg'], files['train_pos']], axis=0, ignore_index=True)
        train_data = pd.concat([train_data, train_review_feature], axis=1, ignore_index=True)
        train_headers = files['header'].values.tolist()[0]
        train_headers.extend(train_review_feature.columns.to_list())
        train_data.columns = train_headers
        train_data = train_data[selected_columns]

        test_data = pd.concat([files['test_neg'], files['test_pos']], axis=0, ignore_index=True)
        test_data = pd.concat([test_data, test_review_feature], axis=1, ignore_index=True)
        test_headers = files['header'].values.tolist()[0]
        test_headers.extend(test_review_feature.columns.to_list())
        test_data.columns = test_headers
        test_data = test_data[selected_columns]

        return train_data, test_data

    def build_features(self):
        train_data, test_data = self.model_data()
        selected_features = ['helpful_votes', 'total_votes', 'negative_score', 'positive_score', 'vine_encoded',
                             'verified_purchase_encoded']
        selected_label = 'target_label'

        train_data['vine_encoded'] = train_data['vine'].apply(lambda x: 1 if x == 'Y' else 0)
        train_data['verified_purchase_encoded'] = train_data['verified_purchase'].apply(lambda x: 1 if x == 'Y' else 0)

        test_data['vine_encoded'] = test_data['vine'].apply(lambda x: 1 if x == 'Y' else 0)
        test_data['verified_purchase_encoded'] = test_data['verified_purchase'].apply(lambda x: 1 if x == 'Y' else 0)

        x_train = train_data[selected_features]
        x_test = test_data[selected_features]
        y_train = train_data[selected_label]
        y_test = test_data[selected_label]

        return x_train, x_test, y_train, y_test

    def build_corpus_id2word_mapping(self):
        train_reviews, train_labels, test_reviews, test_labels = self.__extract_data.process_freq_text()

        clean_list = self.__process_text.lda_processing(train_reviews)

        id2word = corpora.Dictionary(clean_list)
        texts = clean_list
        corpus = [id2word.doc2bow(text) for text in texts]

        return corpus, id2word


