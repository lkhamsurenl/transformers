from transformers import AutoTokenizer, AutoConfig
import torch
from math import sqrt


class Embeddings(torch.nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.positional_embeddings = torch.nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=config.hidden_size)
        self.dropout = torch.nn.Dropout(p=config.attention_probs_dropout_prob)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.size(-1) # size of the input sequence
        position_ids = torch.arange(seq_len, dtype = torch.long).unsqueeze(0)
        token_embeddings = self.token_embeddings(token_ids)
        positional_embeddings = self.positional_embeddings(position_ids)
        embeddings = token_embeddings + positional_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def scaled_dot_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # computed scaled dot attention method
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = torch.nn.functional.softmax(scores, dim = -1)
    return torch.bmm(weights, value)


class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.q = torch.nn.Linear(embed_dim, head_dim)
        self.k = torch.nn.Linear(embed_dim, head_dim)
        self.v = torch.nn.Linear(embed_dim, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scaled_dot_attention(
            self.q(x),
            self.k(x),
            self.v(x)
        )


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            AttentionHead(
                embed_dim = config.hidden_size,
                head_dim = config.hidden_size // config.num_attention_heads
            ) for _ in range(config.num_attention_heads)
        ])
        self.output_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # apply heads on the input hidden state in parallel & concat
        x = torch.cat([h(hidden_state) for h in self.heads], dim = -1)
        x = self.output_layer(x)
        return x


class FNN(torch.nn.Module):
    def __init__(self, embed_dim: int, inter_dim: int):
        super().__init__()
        self.linear_layer_1 = torch.nn.Linear(embed_dim, inter_dim)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout()
        self.linear_layer_2 = torch.nn.Linear(inter_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        x = self.sigmoid(x)
        return x


class AttentionClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding
        self.embeddings = Embeddings(config)
        # multi-head attention
        self.attention_heads = MultiHeadAttention(config)
        # FNN for binary classification
        self.fnn = FNN(config.hidden_size, config.intermediate_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.attention_heads(x)[:, 0, :]
        x = self.fnn(x)[:, 0]
        return x
