import gensim
from utils import common
from gensim.models import CoherenceModel
from utils import common
import numpy as np
import tqdm
import pandas as pd
import math
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


class LDA:

    def __init__(self, corpus, id2word: dict, clean_list: [[str]]):
        self._config = common.read_configs()
        self._corpus = corpus
        self._id2word = id2word
        self._clean_list = clean_list

    def train_lda_model(self, corpus, num_topics: int, random_state: int = 100, chunk_size: int = 100,
                        passes: int = 10, alpha: float = 1, beta: float = 1):
        # num_topics = __config = common.read_configs()['lda']['num_topics']
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=self._id2word,
                                               num_topics=num_topics,
                                               random_state=random_state,
                                               chunksize=chunk_size,
                                               passes=passes,
                                               alpha=alpha,
                                               eta=beta
                                               )

        coherence_model = CoherenceModel(model=lda_model,
                                         texts=self._clean_list,
                                         dictionary=self._id2word,
                                         coherence=self._config['lda']['coherence'])
        coherence_score = coherence_model.get_coherence()

        return lda_model, coherence_score

    def lda_cross_validation(self):

        grid = {'Validation_Set': {}}

        topic_range = range(self._config['lda']['min_topics'], self._config['lda']['max_topics'],
                            self._config['lda']['step_size'])

        alpha = list(np.arange(self._config['lda']['alpha_min'], self._config['lda']['alpha_max'],
                               self._config['lda']['alpha_increment']))
        alpha.append('symmetric')
        alpha.append('asymmetric')

        beta = list(np.arange(self._config['lda']['beta_min'], self._config['lda']['beta_max'],
                              self._config['lda']['beta_increment']))
        beta.append('symmetric')

        # validation_set
        num_of_docs = len(self._corpus)
        corpus_sets = [gensim.utils.ClippedCorpus(self._corpus, math.floor(num_of_docs * 0.75)),
                       gensim.utils.ClippedCorpus(self._corpus, num_of_docs)]
        corpus_title = ['75% Corpus', '100% Corpus']

        model_results = {'validation_set': [],
                         'topics': [],
                         'alpha': [],
                         'beta': [],
                         'coherence': []}

        pbar = tqdm.tqdm(total=540)
        # iterate through validation corpus
        for i in range(len(corpus_sets)):
            for k in topic_range:
                for a in alpha:
                    for b in beta:
                        lda_model, cv = self.train_lda_model(corpus=corpus_sets[i],
                                                             num_topics=k,
                                                             alpha=a,
                                                             beta=b)
                        model_results['validation_set'].append(corpus_title[i])
                        model_results['topics'].append(k)
                        model_results['alpha'].append(a)
                        model_results['beta'].append(b)
                        model_results['coherence'].append(cv)

                        pbar.update(1)

        results_df = pd.DataFrame(model_results)
        pbar.close()

        return results_df

    def lda_vis(self, lda_model):
        pyLDAvis.enable_notebook()
        ldavis_prepared = gensimvis.prepare(lda_model, self._corpus, self._id2word)
        return ldavis_prepared
