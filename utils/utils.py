from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextProcessing:

    def __init__(self):
        self.stopwords_english = stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def process_sentence(self, sentence: str) -> list:
        """
        This function preprocess the data by removing stopwords and lemmatizing
        :param sentence: any sentence
        :return: list of clean lemmatized words
        """
        tokens = word_tokenize(sentence)
        word_list = []
        for _word in tokens:
            if _word not in self.stopwords_english:
                _lemma = self.wordnet_lemmatizer.lemmatize(_word)
                word_list.append(_lemma)
        return word_list
