import gensim
from utils import common


def build_lda_model(corpus, id2word):
    num_topics = __config = common.read_configs()['lda']['num_topics']
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    return lda_model
