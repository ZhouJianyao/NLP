import torch


class Config(object):
    seed = 1234
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'data'
    train_dir = 'senti.train.tsv'
    val_dir = 'senti.dev.tsv'
    test_dir = 'senti.test.tsv'
    batch_size = 64
    dropout_rate = 0.5
    output_size = 1
    embedding_size = 200
    learning_rate = 0.001