import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from transformers import AutoTokenizer, AutoConfig


def scaled_dot_product(query, key, value):
    """
    Produce scaled dot product
    :param query: (batch_size, embed_dim, head_dim)
    :param key: (batch_size, embed_dim, head_dim)
    :param value: (batch_size, embed_dim, head_dim)
    :return: (batch_size, )
    """
    # compute scaled dot product for the transformer
    dim_k = key.size(-1)
    # mat-mul query & key. (batch_size, embed_dim, embed_dim)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    # scale the scores
    weights = F.softmax(scores, dim = -1)
    # mat-mul weights & value. (batch_size, embed_dim, head_dim)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """
        Initialize attention head
        :param embed_dim: Size of the embedding
        :param head_dim: Size of the head
        """
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state)
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim = -1)
        x = self.output_layer(x)
        return x


class Feedforward(nn.Module):
    def __init__(self, embed_dim: int, inter_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, inter_dim)
        self.linear2 = nn.Linear(inter_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int, inter_dim: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim=embed_dim, head_dim=head_dim, num_heads=num_heads)
        self.feedforward = Feedforward(embed_dim=embed_dim, inter_dim=inter_dim)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        # attention + skip layer
        x = x + self.attention(hidden_state)
        # FF + skip layer
        x = x + self.feedforward(self.layer_norm_2(x))
        return x


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_position_embeddings: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # input_ids are tokens that have not yet been mapped to embeddings
        seq_len = input_ids.size(-1)
        position_ids = torch.arange(seq_len, dtype = torch.long).unsqueeze(0)
        token_embeddings= self.token_embeddings(input_ids)
        position_embeddings = self.positional_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            head_dim: int,
            num_heads: int,
            inter_dim: int,
            vocab_size: int,
            max_position_embeddings: int,
            num_hidden_layers: int
    ):
        super().__init__()
        self.embedding = Embeddings(vocab_size=vocab_size, embed_dim=embed_dim, max_position_embeddings=max_position_embeddings)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                inter_dim=inter_dim
            ) for _ in range(num_hidden_layers)
        ])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.encoder:
            x = layer(x)
        return x


class TransformerForSequenceClassification(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            head_dim: int,
            num_heads: int,
            inter_dim: int,
            vocab_size: int,
            max_position_embeddings: int,
            num_hidden_layers: int
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            inter_dim=inter_dim,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=num_hidden_layers
        )
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(embed_dim, 3)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]  # select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x


MODEL_CHKP = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHKP)
config = AutoConfig.from_pretrained(MODEL_CHKP)
text = "I love cheese!"
inputs = tokenizer(text, return_tensors = "pt", add_special_tokens = False)
model = TransformerForSequenceClassification(
    embed_dim=config.hidden_size,
    head_dim=config.hidden_size // config.num_attention_heads,
    num_heads=config.num_attention_heads,
    inter_dim = config.intermediate_size,
    vocab_size=config.vocab_size,
    max_position_embeddings=config.max_position_embeddings,
    num_hidden_layers=config.num_hidden_layers,
)
outputs = model(inputs.input_ids)

print(f"""
embed_dim = {config.hidden_size}
num_heads = {config.num_attention_heads}
ff_outputs = {outputs}
""")

