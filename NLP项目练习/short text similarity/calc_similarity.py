#Author ZhouJY

import GlobalParament
import utils
from gensim.models import word2vec


def calc_sim(model_dir, train_dir, test_dir, result_output_dir):
    model = word2vec.Word2Vec.load(model_dir)
    vocab = set(model.wv.vocab.keys())
    # 将相似度结果写入的文件
    f_writer = open(result_output_dir, 'w', encoding=GlobalParament.encoding)

    with open(test_dir, 'r', encoding=GlobalParament.encoding) as f_test_reader:
        f_test = f_test_reader.readlines()
        for i, test_line in enumerate(f_test):
            test_line = utils.delete_r_n(test_line)
            test_id, test_content = test_line.split(',')
            print('测试文段：' + test_content)

            test_content_list = test_content.split()
            test_content_list = utils.clear_word_from_vocab(test_content_list, vocab)

            sim_score = {}

            with open(train_dir, 'r', encoding=GlobalParament.encoding) as f_train_reader:
                f_train = f_train_reader.readlines()
                for train_line in f_train:
                    train_line = utils.delete_r_n(train_line)
                    train_id, train_content = train_line.split(',')
                    train_content_list = train_content.split()
                    train_content_list = utils.clear_word_from_vocab(train_content_list, vocab)

                    if len(train_content_list) > 0:
                        sim_score[train_id] = model.wv.n_similarity(test_content_list, train_content_list)

            sim_score = sorted(sim_score.items(), key=lambda x: x[1], reverse=True)
            print('开始记录前10个最相似的文段')
            train_result_ids = sim_score[0][0]
            for id, _ in sim_score[1: 10]:
                train_result_ids += ' ' + id
            f_writer.write(test_id + ',' + train_result_ids + '\n')
            f_writer.flush()
            print(f'处理完第{i+1}条测试数据')
    f_writer.close()


if __name__ == '__main__':
    calc_sim(GlobalParament.model_output_path, GlobalParament.train_after_process_text_dir,
             GlobalParament.test_after_process_text_dir, GlobalParament.result_output_path)
