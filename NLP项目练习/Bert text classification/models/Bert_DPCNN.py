import torch
from torch import nn
from pytorch_pretrained import BertModel, BertTokenizer
from pathlib import Path


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'Bert_DPCNN'
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
        # 卷积核数量
        self.num_filters = 250
        # dropout率
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size))
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1))
        self.max_pool = nn.MaxPool2d((3, 1), stride=2)

        # 四个参数分别表示左右上下，在上下各填充一行
        self.pad1 = nn.ZeroPad2d((0, 0, 1, 1))
        # 四个参数分别表示左右上下，在下填充一行
        self.pad2 = nn.ZeroPad2d((0, 0, 0, 1))

        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def _block(self, x):
        x = self.pad2(x)  # [batch_size, num_filters, 31, 1]
        px = self.max_pool(x)  # [batch_size, num_filters, 15, 1]
        x = self.pad1(px)  # [batch_size, num_filters, 17, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, 15, 1]
        x = self.pad1(x)  # [batch_size, num_filters, 17, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, 15, 1]
        x = x + px

        return x

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        out = self.conv_region(out)  # [batch_size, num_filters, 30, 1]
        out = self.pad1(out)  # [batch_size, num_filters, 32, 1]
        out - self.relu(out)
        out = self.conv(out)  # [batch_size, num_filters, 30, 1]
        out = self.pad1(out)  # [batch_size, num_filters, 32, 1]
        out - self.relu(out)
        out = self.conv(out)  # [batch_size, num_filters, 30, 1]
        out - self.relu(out)
        while out.size()[2] > 2:
            out = self._block(out)
        out = out.squeeze()
        out = self.fc(out)

        return out


"/Users/zhoujianyao/opt/anaconda3/bin/python"