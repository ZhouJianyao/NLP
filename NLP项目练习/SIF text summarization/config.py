from utils import get_stopwords
from gensim.models.word2vec import Word2Vec


class Config(object):
    def __init__(self):
        self.stopwords = get_stopwords(path='data/stopwords.txt')
        self.model = Word2Vec.load('model/word2vec_v1.0.model')
        self.a = 1e-3
        self.unlisted_word_freq = 0.0001
        self.knn_n = (1, 1)
        self.knn_weights = [1 / 3, 1 / 3, 1 / 3]
        self.top_n = "auto"  # 可取int，也可以取"auto"
        self.remain_sentence_order = True
