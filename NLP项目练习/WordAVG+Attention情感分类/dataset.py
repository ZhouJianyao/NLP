import torch
from torchtext.data import Field, TabularDataset
from utils import tokenizer
from config import Config

config = Config()


def load_dataset(path, train_dir, val_dir, test_dir):
    TEXT = Field(tokenize=tokenizer, batch_first=True, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

    train, val, test = TabularDataset.splits(
        path=path,
        train=train_dir,
        validation=val_dir,
        test=test_dir,
        format='tsv',
        fields=[('text', TEXT), ('label', LABEL)]
    )
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    return train, val, test, TEXT, LABEL
