# -*- coding: utf-8 -*-

import logging, jieba, os, re
from tqdm import tqdm


def split_sentence(document):
    '''
    this is to split a doc into sentences
    分句之前先把\n，\r，\r\n，\u3000直接洗掉即可
    分句的规则有：1.碰到句号，问号，感叹号，分号还有空格就分，并把标点符号加到前句
    '''
    document = document.strip().replace("\r", "").replace("\n", "").replace("\\n", "").replace("\u3000", "").replace("\\", "")
    # 以结尾标点切分
    split = re.split(r"([？?。!！；…])", document)
    # 标点符号加回句子
    split.append("")
    split = [i+j for (i, j) in zip(split[0::2], split[1::2])]
    return split


def wash_and_split(input_text):
    '''
    主要的数据预处理函数
    输入一篇文章string
    输出分词分句并清洗后的句子列表
    '''

    sentences = split_sentence(input_text)
    cut = list(map(jieba.lcut, sentences))

    stop_words = get_stopwords(path='data/stopwords.txt')

    def remove_stopwords_for_each_sentence(list_of_words):
        return [word for word in list_of_words if word not in stop_words and word != ' ']

    cut = list(map(remove_stopwords_for_each_sentence, cut))

    return [i for i in cut if i not in [[], ["…"]]]


def parse_wiki(file_path, output_path):
    # 过滤掉<doc>
    regex_str = "[^<doc.*>$]|[^</doc>$]"
    with open(file_path, "r", encoding="utf-8") as f_read:
        # 写文件
        with open(output_path, "w+", encoding="utf-8") as f_write:
            content_line = f_read.readline()
            # 获取停用词表
            stopwords = get_stopwords()
            # 定义一个字符串变量，表示一篇文章的分词结果
            article_contents = ""
            while content_line:
                match_obj = re.match(regex_str, content_line)
                content_line = content_line.strip("\n")
                if len(content_line) > 0:
                    if match_obj:
                        words = jieba.cut(content_line, cut_all=False)
                        for word in words:
                            if word not in stopwords:
                                article_contents += word + " "
                    else:
                        if len(article_contents) > 0:
                            f_write.write(article_contents + "\n")
                            article_contents = ""
                content_line = f_read.readline()


def generate_corpus():
    wiki_path = "data/AA"
    save_path = "data/output"
    for i in range(3):
        file_path = os.path.join(wiki_path, str("wiki_0%s" % str(i)))
        parse_wiki(file_path, os.path.join(save_path, "wiki_corpus0%s" % str(i)))
    print(f"Wiki Data Saved into: {save_path}")


def merge_corpus():
    '''
    合并分词后的文件
    '''
    output_path = open("data/output/wiki_sum", "w", encoding="utf-8")
    input_path = "data/output"
    for i in tqdm(range(3)):
        file_path = os.path.join(input_path, str("wiki_corpus0%s" % str(i)))
        with open(file_path, "r", encoding="utf-8") as f_read:
            line = f_read.readline()
            while line:
                output_path.writelines(line)
                line = f_read.readline()
    output_path.close()


def get_stopwords(path='data/stopwords.txt'):
    # 加载停用词表
    stopwords_set = []
    with open(path, 'r', encoding="utf-8") as stopwords:
        for stopword in stopwords:
            if stopword != '':
                stopwords_set.append(stopword.strip("\n").strip())
    stopwords_set = set(stopwords_set)
    return stopwords_set


def remove_stopwords(words):
    return [word.strip() for word in words if word not in get_stopwords() and word != " "]


if __name__ == "__main__":
    # generate_corpus()
    # merge_corpus()

    with open('data/new_data_to_train_word2vec.txt', 'w', encoding='utf-8') as f_write:
        with open("data/output/wiki_sum", "r", encoding="utf-8") as wiki_data:
            print('开始写入wiki数据')
            count1 = 1
            line = wiki_data.readline()
            while line:
                line = line.strip().split(' ')
                line = remove_stopwords(line)
                # a = list(map(remove_stopwords, [line.strip().split(' ') for line in wiki_data]))
                f_write.write(" ".join(line) + '\n')
                if count1 % 100 == 0:
                   print(f'完成{count1}条数据')
                line = wiki_data.readline()
                count1 += 1
            print('完成写入所有数据')


