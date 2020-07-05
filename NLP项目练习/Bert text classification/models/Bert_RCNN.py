#Author ZhouJY

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'Bert_RCNN'
        # 训练集
        self.train_path = Path(dataset).joinpath('data/train.txt')
        # 验证集
        self.dev_path = Path(dataset).joinpath('data/dev.txt')
        # 测试集
        self.test_path = Path(dataset).joinpath('data/test.txt')
        # dataset
        self.dataset_pkl = Path(dataset).joinpath('data/dataset.pkl')
        # 类别
        self.class_list = [c.strip() for c in Path(dataset).joinpath('data/class.txt').read_text().split('\n')]
        # 模型保存路径
        self.save_path = Path(dataset).joinpath('save_dict', self.model_name + '.ckpt')
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 早停数
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
        self.bert_path = './bert_pretrain'
        # 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 隐层数
        self.hidden_size = 768
        # RNN隐层数
        self.rnn_hidden = 256
        # RNN层数
        self.rnn_layers = 2
        # dropout率
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.rnn_layers,
                                bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)

    def forward(self, x):
        # x: [ids, seq_len, mask]
        context = x[0]  # 输入的句子字 shape：[128, 32]
        mask = x[2]  # 对padding部分的mask shape: [128, 32]
        # out shape: [128, 32, 768]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        # 此时维度为[128, 32, 256*2]，需要对第二维做maxpool，若直接maxpool则会对第三维做maxpool，所以要先交换第二和第三维度
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out)
        # 此时维度为[128, 256*2, 1]，需要将最后一维去掉
        out = out.squeeze(-1)
        out = self.fc(out)

        return out
