#Author ZhouJY

import GlobalParament
import jieba
import pandas as pd
import re
from gensim.models import word2vec

# 去除回车换行
def delete_r_n(text):
    line = text.replace('\r', '').replace('\n', '').strip()
    return line

# 读取停用词
def get_stopwords(stopwords_dir):
    stopwords = []

    with open(stopwords_dir, 'r', encoding=GlobalParament.encoding) as f:
        for line in f:

            line = delete_r_n(line)
            stopwords.append(line)

    stopwords = set(stopwords)

    return stopwords


# jieba精确分词
def jieba_cut(content, stopwords):
    word_list = []

    if content != '' and content is not None:
        seg_list = jieba.cut(content.replace(' ', ''))
        for word in seg_list:
            if word not in stopwords:
                word_list.append(word)

    else:
        raise ValueError

    return word_list


# jieba搜索引擎分词
def jieba_cut_for_search(content, stopwords):
    word_list = []

    if content != '' and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            if word not in stopwords:
                word_list.append(word)

    else:
        raise ValueError

    return word_list


# 消除不在词汇表中的词
def clear_word_from_vocab(word_list, vocab):
    new_word_list = []

    for word in word_list:
        if word in vocab:
            new_word_list.append(word)

    return new_word_list


# pandas文本预处理
def preprocessing_text_pd(text_dir, after_process_text_dir, stop_words_dir):
    stopwords = get_stopwords(stop_words_dir)
    sentences = []
    df = pd.read_csv(text_dir)

    for index, row in df.iterrows():
        if (index+1) % 10000 == 0:
            print(f'finish {index}/{len(df)} sentences.')

        title = delete_r_n(row['title'])
        word_list = jieba_cut(title, stopwords)
        df.loc[index, 'title'] = ' '.join(word_list)
        sentences.append(list(word_list))

    df.to_csv(after_process_text_dir, encoding=GlobalParament.encoding, index=False)

    return sentences


# 优化文本预处理
def preprocessing_text(text_dir, after_process_text_dir, stop_words_dir):
    stopwords = get_stopwords(stop_words_dir)
    sentences = []

    with open(after_process_text_dir, 'w', encoding=GlobalParament.encoding) as f_writer:
        with open(text_dir, 'r', encoding=GlobalParament.encoding) as f_reader:
            for i, line in enumerate(f_reader):
                if i == 0:
                    continue
                if (i + 1) % 10000 == 0:
                    print(f'finish {i+1} sentences.')
                id, sentence = re.split(pattern=',', string=line, maxsplit=1)
                sentence = jieba_cut(delete_r_n(sentence), stopwords)
                sentences.append(sentence)
                f_writer.write(id + ',' + ' '.join(sentence) + '\n')

    return sentences


if __name__ == '__main__':
    # stop_words = get_stopwords(GlobalParament.stop_word_dir)

    preprocessing_text(GlobalParament.test_set_dir,
                    GlobalParament.test_after_process_text_dir, GlobalParament.stop_word_dir)



