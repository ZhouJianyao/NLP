import argparse
import time
import torch
from torch import nn
from torch import optim
from torchtext.data import BucketIterator
from dataset import load_dataset
from config import Config
from WordAVGAttention import WordAVGAttention
from train import train, evaluate
from utils import epoch_time


config = Config()
parser = argparse.ArgumentParser()
parser.add_argument('--epoches', default=10, type=int, help='training epoches')
args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    device = config.device

    print('开始处理数据')
    # 加载数据
    train_data, val_data, test_data, TEXT, LABEL = load_dataset(config.path,
                                                                config.train_dir,
                                                                config.val_dir,
                                                                config.test_dir)
    train_iter, val_iter, test_iter = BucketIterator.splits((train_data, val_data, test_data),
                                       batch_size=config.batch_size,
                                       sort_key=lambda x: len(x.text),
                                       sort_within_batch=True,
                                       repeat=False,
                                       device=config.device)
    vocab_size = len(TEXT.vocab)
    pad_idx = TEXT.vocab.stoi['<pad>']
    model = WordAVGAttention(vocab_size, config.embedding_size, config.dropout_rate, config.output_size, pad_idx)
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    N_epoches = args.epoches
    best_val_loss = float('inf')
    for epoch in range(N_epoches):
        start_time = time.time()

        train_acc, train_loss = train(model, train_iter, TEXT, criterion, optimizer)
        val_acc, val_loss = evaluate(model, val_iter, TEXT, criterion)

        end_time = time.time() - start_time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_val_loss:
            print('This epoch loss increasing')
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model/word_avg_attention.pth')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')

