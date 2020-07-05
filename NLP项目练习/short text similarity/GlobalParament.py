#Author ZhouJY

encoding = 'utf-8'  # 编码设置
stop_word_dir = 'data/stop_words.txt'  # 停用词路径
train_set_dir = 'data/train.csv'  # 训练集路径
test_set_dir = 'data/test.csv'  # 测试集路径
train_after_process_text_dir = 'data/train_after_process.csv'  # 训练集预处理后路径
test_after_process_text_dir = 'data/test_after_process.csv'  # 测试集预处理后路径
model_output_path = 'model/word2vec_news.model'  # 模型输出路径
result_output_path = 'data/sim_result.csv'  # 相似度结果路径
similarity_out_path = 'data/sim_out.txt'  # 相似文档结果路径

# word2vec参数
train_size = 150
train_window = 5
