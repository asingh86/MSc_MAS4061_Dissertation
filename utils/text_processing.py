import pandas as pd
import numpy as np
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import common

# download necessary files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TextProcessing:

    def __init__(self):
        self._config = common.read_configs()
        self.__stopwords = stopwords.words('english')
        self.__punctuations = string.punctuation
        self.__wordnet_lemmatizer = WordNetLemmatizer()

    @staticmethod
    def lowercase(text: list) -> list:
        return [_word.lower() for _word in text]

    @staticmethod
    def sentence_tokenizer(text: str) -> list:
        """Tokenize a sentences and converts into lowercase"""
        return [_word.lower() for _word in word_tokenize(text)]

    def remove_stopwords(self, text: list) -> list:
        return [_word for _word in text if _word not in self.__stopwords]

    def remove_manual_stopwords(self, text: list) -> list:
        return [_word for _word in text if _word not in self._config['filters']['manual_stopwords_list']]

    def remove_punctuations(self, text: list) -> list:
        return [_word for _word in text if _word not in self.__punctuations]

    def text_lemmatizer(self, text: list) -> list:
        return [self.__wordnet_lemmatizer.lemmatize(_word) for _word in text]

    def process_text(self, text: str) -> list:
        processed_text = self.sentence_tokenizer(str(text))
        if self._config['filters']['lowercase']:
            processed_text = self.lowercase(processed_text)
        if self._config['filters']['stopwords']:
            processed_text = self.remove_stopwords(processed_text)
        if self._config['filters']['manual_stopwords']:
            processed_text = self.remove_manual_stopwords(processed_text)
        if self._config['filters']['punctuation']:
            processed_text = self.remove_punctuations(processed_text)
        if self._config['filters']['lemmatize']:
            processed_text = self.text_lemmatizer(processed_text)

        return processed_text

    @staticmethod
    def isNaN(text: str) -> bool:
        return string != string

    def build_freq(self, sentences: [str], ys: [int]) -> dict:

        freq = {}
        for sentence, y in zip(sentences, ys):
            if not self.isNaN(sentence):
                for _word in self.process_text(sentence,):
                    pair = (_word, y)
                    freq[pair] = freq.get(pair, 0) + 1
        return freq

    def word_count_feature_extraction(self, sentences: str, ys: list, word_freq: dict):

        m = len(sentences)
        x = np.zeros((m, 3))

        for i in range(m):
            neg = 0
            pos = 0
            if not self.isNaN(sentences[i]):
                for word in list(set(self.process_text(sentences[i]))):
                    if (word, 1) in word_freq:
                        neg += word_freq[(word, 1)]
                    if (word, 0) in word_freq:
                        pos += word_freq[(word, 0)]
            x[i, :] = [neg, pos, ys[i]]
        return x

