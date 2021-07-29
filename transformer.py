import pandas as pd
from utils.extractor import DataExtractor
from utils.text_processing import TextProcessing
from category_encoders import TargetEncoder


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

        encoder = TargetEncoder()
        train_data['product_category_encoded'] = encoder.fit_transform(train_data['product_category'],
                                                                       train_data['target_label'])
        train_data['vine_encoded'] = train_data['vine'].apply(lambda x: 1 if x == 'Y' else 0)
        train_data['verified_purchase_encoded'] = train_data['verified_purchase'].apply(lambda x: 1 if x == 'Y' else 0)

        test_data[]