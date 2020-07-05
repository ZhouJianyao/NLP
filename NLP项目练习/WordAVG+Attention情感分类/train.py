import torch
from config import Config
from utils import binary_accuracy

config = Config()

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
device = config.device


def train(model, train_iter, TEXT, criterion, optimizer):
    epoch_acc = 0.
    epoch_loss = 0.
    model.train()

    for batch in train_iter:
        x, _ = batch.text
        y = batch.label
        mask = 1 - (x==TEXT.vocab.stoi['<pad>']).float()
        score = model(x, mask)
        loss = criterion(score, y)
        acc = binary_accuracy(score, y)
        epoch_acc += acc.item()
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_acc/len(train_iter), epoch_loss/len(train_iter)


def evaluate(model, val_iter, TEXT, criterion):
    epoch_acc = 0.
    epoch_loss = 0.
    model.eval()

    with torch.no_grad():
        for batch in val_iter:
            x, _ = batch.text
            y = batch.label
            mask = 1 - (x==TEXT.vocab.stoi['<pad>']).float()
            score = model(x, mask)
            loss = criterion(score, y)
            acc = binary_accuracy(score, y)
            epoch_acc += acc.item()
            epoch_loss += loss.item()

        return epoch_acc/len(val_iter), epoch_loss/len(val_iter)
