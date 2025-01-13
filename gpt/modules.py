import torch
from typing import Dict


class Embeddings(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["embed_dim"])
        self.position_embeddings = torch.nn.Embedding(num_embeddings=config["context_length"],
                                                   embedding_dim=config["embed_dim"])
        self.dropout = torch.nn.Dropout(config["drop_rate"])

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """

        :param input_tokens: (batch_size, seq_size)
        :return: (batch_size, seq_size, embed_dim)
        """
        seq_size = input_tokens.size(-1)
        position_ids = torch.arange(seq_size).unsqueeze(0)
        token_emb = self.token_embeddings(input_tokens)
        position_emb = self.position_embeddings(position_ids)
        emb = token_emb + position_emb
        emb = self.dropout(emb)
        return emb


def scaled_dot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """

    :param query: (batch_size, seq_size, head_dim)
    :param key:
    :param value:
    :param mask: (context_length, context_length) upper triangular matrix
    :return: (batch_size, seq_size, head_dim)
    """
    batch_size, seq_size, head_dim = key.size()
    scores = torch.bmm(query, key.transpose(1, 2))
    scores = scores.masked_fill(mask.bool()[:seq_size, :seq_size], -torch.inf)
    weights = torch.softmax(scores / head_dim**0.5, dim = -1)
    return torch.bmm(weights, value)


class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, context_length: int):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, head_dim)
        self.key = torch.nn.Linear(embed_dim, head_dim)
        self.value = torch.nn.Linear(embed_dim, head_dim)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: (batch_size, seq_size, embed_dim)
        :return: (batch_size, seq_size, head_dim)
        """
        return scaled_dot_attention(
            self.query(x),
            self.key(x),
            self.value(x),
            self.mask,
        )


class MultiAttentionHeads(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        embed_dim = config["embed_dim"]
        head_dim = config["embed_dim"] // config["n_heads"]
        self.heads = torch.nn.ModuleList([
            AttentionHead(embed_dim, head_dim, config["context_length"]) for _ in range(config["n_heads"])
        ])
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: (batch_size, seq_size, embed_dim)
        :return: (batch_size, seq_size, embed_dim)
        """
        x = torch.concat([h(x) for h in self.heads], dim = -1)
        x = self.out_proj(x)
        return x


class FeedForwardNet(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config["embed_dim"], 4 * config["embed_dim"]),
            torch.nn.GELU(),
            torch.nn.Linear(4 * config["embed_dim"], config["embed_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_size, embed_dim)
        :return: (batch_size, seq_size, embed_dim)
        """
        return self.layers(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(config["embed_dim"])
        self.attn = MultiAttentionHeads(config)
        self.norm2 = torch.nn.LayerNorm(config["embed_dim"])
        self.fnn = FeedForwardNet(config)
        self.dropout = torch.nn.Dropout(config["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: (batch_size, seq_size, embed_dim)
        :return: (batch_size, seq_size, embed_dim)
        """
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.fnn(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class GPTModel(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Embeddings(config),
            *[TransformerBlock(config) for _ in range(config["n_layers"])],
            torch.nn.LayerNorm(config["embed_dim"]),
            torch.nn.Linear(config["embed_dim"], config["vocab_size"])
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """

        :param token_ids: (batch_size, seq_size)
        :return: (batch_size, seq_size, vocab_size)
        """
        return self.layers(token_ids)
