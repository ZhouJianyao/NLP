#Author ZhouJY

import GlobalParament
import utils
from gensim.models import word2vec


# 训练word2vec
def train(sentences, model_out_put_path):
    print('开始训练')
    model = word2vec.Word2Vec(sentences=sentences, size=GlobalParament.train_size,
                              window=GlobalParament.train_window, min_count=20)
    model.save(model_out_put_path)
    print('训练完成')


if __name__ == '__main__':
    # stop_words = utils.get_stopwords(GlobalParament.stop_word_dir)
    content = utils.preprocessing_text(GlobalParament.train_set_dir,
                                         GlobalParament.train_after_process_text_dir, GlobalParament.stop_word_dir)
    train(content, GlobalParament.model_output_path)
    model = word2vec.Word2Vec.load(GlobalParament.model_output_path)
    vocab = list(model.wv.vocab.keys())
    print(len(vocab))
