
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
from pathlib import Path


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        # 训练集
        self.train_path = Path(dataset).joinpath('data/train.txt')
        # 测试集
        self.test_path = Path(dataset).joinpath('data/test.txt')
        # 校验集
        self.dev_path = Path(dataset).joinpath('data/dev.txt')
        # dataset
        self.dataset_pkl = Path(dataset).joinpath('data/dataset.pkl')
        # 类别
        self.class_list = [c.strip() for c in Path(dataset).joinpath('data/class.txt').read_text().split('\n')]
        # 模型训练结果
        self.save_path = Path(dataset).joinpath('save_dict', self.model_name + '.ckpt')
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 提前结束训练
        self.require_improvement = 1000
        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.epochs = 3
        # batch尺寸
        self.batch_size = 128
        # pad尺寸
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练路径
        self.bert_path = './ERNIE_pretrain'
        # 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 隐层数
        self.hidden_size = 768
        # dropout率
        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[128,32]
        mask = x[2]  # 对padding部分进行mask shape[128,32]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # shape [128,768]
        out = self.fc(pooled)  # shape [128,10]
        return out

























