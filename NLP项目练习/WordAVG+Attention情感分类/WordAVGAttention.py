import torch
from torch import nn
import torch.nn.functional as F


class WordAVGAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout_rate, output_size, pad_idx):
        super(WordAVGAttention, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(embedding_size, output_size)
        # K权重矩阵
        self.coef = nn.Parameter(torch.randn(embedding_size))
        self.init_weights()

    def init_weights(self):
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, mask):
        """
        x: [batch_size, seq_len]
        mask: [batch_size, seq_len]
        """
        # [batch_size, seq_len, embedding_size]
        embed = self.embeddings(x)
        context, _ = self.attention(embed, self.coef, embed, mask=mask)

        return self.linear(context).squeeze()

    def attention(self, query, key, value, mask=None):
        key = key.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_size]
        scores = F.cosine_similarity(query, key, dim=-1)
        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)
        p_attention = F.softmax(scores, dim=-1).unsqueeze(1)  # [batch_size, 1, seq_len]

        return torch.bmm(p_attention, value).squeeze(), p_attention


