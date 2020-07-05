# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from gensim.models import word2vec
import jieba
from config import Config

config = Config()


# convert a list of sentence with word2vec items into a set of sentence vectors
def SIF(sentence_list, model, a=1e-3, unlisted_word_freq=0.0001):
    '''
    Input:
        sentence_list: a list of tokenized sentences, text format
        a: param
        model: word2vec model object, pretrained by gensim
        embedding_size: word2vec model embedding size
    Output:
        SIF sentence embeddings
    '''
    
    v_lookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    embedding_size = model.vector_size  # Embedding size

    vocab_count = 0
    for k in v_lookup:
        vocab_count += v_lookup[k].count  # ALL Words Count

    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        for word in sentence:
            if word in v_lookup:
                a_value = a / (a + v_lookup[word].count / vocab_count)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, vectors[word]))  # vs += sif * word_vector
            else:
                a_value = a / (a + unlisted_word_freq)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, np.zeros(embedding_size)))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average

        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.dot(u, np.transpose(u))  # u x uT

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = [vs - vs*u for vs in sentence_set]

    return np.array(sentence_vecs)


if __name__ == "__main__":
    model = config.model
    
    sample_text = ["今天小明吃了一个包子，他觉得非常好吃。", "于是他又去买了一个包子，然后喂了狗。", "但是还不够爽，于是他又买了一笼包子。", "然后又喂了猫。"]
    sampt = [jieba.lcut(s) for s in sample_text]
    a = SIF(sampt, model)
    print(a)
    print(a.shape)
