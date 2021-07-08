import pandas as pd
import numpy as np
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download necessary files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TextProcessing:

    def __init__(self):
        self.__stopwords = stopwords.words('english')
        self.__punctuations = string.punctuation
        self.__wordnet_lemmatizer = WordNetLemmatizer()

    @staticmethod
    def sentence_tokenizer(text: str) -> list:
        """Tokenize a sentences and converts into lowercase"""
        return [_word.lower() for _word in word_tokenize(text)]

    def remove_stopwords(self, text: list) -> list:
        return [_word for _word in text if _word not in self.__stopwords]

    def remove_punctuations(self, text: list) -> list:
        return [_word for _word in text if _word not in self.__punctuations]

    def text_lemmatizer(self, text: list) -> list:
        return [self.__wordnet_lemmatizer.lemmatize(_word) for _word in text]

    def process_text(self, text: str) -> list:
        processed_text = self.sentence_tokenizer(str(text))
        processed_text = self.remove_stopwords(processed_text)
        processed_text = self.remove_punctuations(processed_text)
        processed_text = self.text_lemmatizer(processed_text)

        return processed_text

    def build_freq(self, sentences: [str], ys: [int]) -> dict:
        def isNaN(text: str) -> bool:
            return string != string

        freq = {}
        for sentence, y in zip(sentences, ys):
            if not isNaN(sentence):
                for _word in self.process_text(sentence):
                    pair = (_word, y)
                    if pair in freq:
                        freq[pair] += 1
                    else:
                        freq[pair] = 1
        return freq
