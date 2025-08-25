from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel
)
import torch

base_config = AutoConfig.from_pretrained("gpt2")

# TODO: Remove once done
# print(f"base_config = {base_config}")

class GPTConfig(PretrainedConfig):
    model_type = "gpt2"

    def __init__(
        self,
        vocab_size: int = base_config.vocab_size,
        n_embd: int = base_config.n_embd,
        n_head: int = base_config.n_head,
        n_layer: int = base_config.n_layer,
        n_ctx: int = base_config.n_ctx,
        n_positions: int = base_config.n_positions,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        super().__init__(**kwargs)


class Embedding(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embd = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd)
        self.position_embd = torch.nn.Embedding(num_embeddings=config.n_positions, embedding_dim=config.n_embd)
        self.dropout = torch.nn.Dropout()

    def forward(self, input_ids: torch.Tensor):
        """
        :param input_ids: (batch_size, seq_size)
        :return: (batch_size, seq_size, n_embd)
        """
        (batch_size, seq_size) = input_ids.size()
        position_ids = torch.arange(seq_size).unsqueeze(0)
        token_emb = self.token_embd(input_ids)
        pos_emb = self.position_embd(position_ids)
        emb = token_emb + pos_emb
        emb = self.dropout(emb)
        return emb


def scaled_dot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """

    :param query: (batch_size, seq_size. head_dim)
    :param key: (batch_size, seq_size, head_dim)
    :param value: (batch_size, seq_size. head_dim)
    :param attention_mask: (n_ctx, n_ctx)
    :return: (batch_size, seq_size, head_dim)
    """
    (batch_size, seq_size, head_dim) = key.size()
    scores = torch.bmm(query, key.transpose(1, 2)) # (batch_size, seq_size, seq_size)
    scores = scores.masked_fill(attention_mask.bool()[:seq_size, :seq_size], value=-torch.inf)
    weight = torch.nn.functional.softmax(scores / seq_size ** 0.5, dim=-1)
    return torch.bmm(weight, value)  # (batch_size, n_ctx, head_dim)


class AttentionHead(torch.nn.Module):
    def __init__(self, n_embd: int, head_dim: int, n_ctx: int):
        super().__init__()
        self.query = torch.nn.Linear(n_embd, head_dim)
        self.key = torch.nn.Linear(n_embd, head_dim)
        self.value = torch.nn.Linear(n_embd, head_dim)
        self.register_buffer(
            "attention_mask",
            torch.triu(torch.ones(n_ctx, n_ctx), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: (batch_size, seq_size, n_embd)
        :return: (batch_size, seq_size, n_embd)
        """
        return scaled_dot_attention(
            self.query(x),
            self.key(x),
            self.value(x),
            self.attention_mask
        )

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_dim = config.n_embd // config.n_head
        self.heads = torch.nn.ModuleList([
            AttentionHead(config.n_embd, head_dim, config.n_ctx) for _ in range(config.n_head)
        ])
        self.output_proj = torch.nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor):
        """
        :param x: (batch_size, seq_size, n_embd)
        :return: (batch_size, seq_size, n_embd)
        """
        hidden = torch.concat([h(x) for h in self.heads], dim=-1)
        return self.output_proj(hidden)


class FNN(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(4 * config.n_embd, config.n_embd),
            torch.nn.Dropout()
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: (batch_size, seq_size, n_embd)
        :return: (batch_size, seq_size, n_embd)
        """
        return self.layers(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm = torch.nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.dropout = torch.nn.Dropout()
        self.fnn = FNN(config)

    def forward(self, x: torch.Tensor):
        """
        :param x: (batch_size, seq_size, n_embd)
        :return: (batch_size, seq_size, n_embd)
        """
        skip = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + skip

        skip = x
        x = self.norm(x)
        x = self.fnn(x)
        x = self.dropout(x)
        x = x + skip

        return x



class GPT2(PreTrainedModel):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.layers = torch.nn.Sequential(
            Embedding(config),
            TransformerBlock(config),
            torch.nn.Linear(config.n_embd, config.vocab_size)
        )

    def forward(self, input_ids: torch.Tensor, labels = None, attention_mask = None):
        logits = self.layers(input_ids)  # (batch_size, seq_size, vocab_size)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1))
            print(f"logits = {logits}; loss = {loss}")
            return {
                "logits": logits,
                "loss": loss,
            }
        return {
            "logits": logits,
        }