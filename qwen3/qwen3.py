import torch 
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    TrainingArguments, 
    Trainer, 
    PreTrainedModel, 
    PretrainedConfig,
    DataCollatorWithPadding
)
from typing import Dict

## 
# MODEL
## 

QWEN3_CONFIG = {
    "vocab_size": 151_936,           # Vocabulary size
    "context_length": 1024,          # Context length that was used to train the model. TODO: Actual value is 40_960
    "emb_dim": 1024,                 # Embedding dimension
    "n_heads": 16,                   # Number of attention heads
    "n_layers": 4,                   # Number of layers. In actuality it's 28
    "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
    "head_dim": 128,                 # Size of the heads in GQA
    "qk_norm": True,                 # Whether to normalize queries and values in GQA
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
}

class QwenDataset(torch.utils.data.Dataset):
    # custom dataset
    def __init__(self, text: str, tokenizer, context_length: int):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text, padding=True, truncation=True)
        for i in range(0, len(token_ids), context_length):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + 1 + context_length]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.target_ids[index]
        }

class Qwen3Config(PretrainedConfig):
    model_type = "qwen3"

    def __init__(
        self, 
        vocab_size: int = QWEN3_CONFIG["vocab_size"],
        context_length: int = QWEN3_CONFIG["context_length"],
        emb_dim: int = QWEN3_CONFIG["emb_dim"],
        n_heads: int = QWEN3_CONFIG["n_heads"],
        n_layers: int = QWEN3_CONFIG["n_layers"],
        hidden_dim: int = QWEN3_CONFIG["hidden_dim"],
        head_dim: int = QWEN3_CONFIG["head_dim"],
        qk_norm: bool = QWEN3_CONFIG["qk_norm"],
        n_kv_groups: int = QWEN3_CONFIG["n_kv_groups"],
        rope_base: float = QWEN3_CONFIG["rope_base"],
        dtype = QWEN3_CONFIG["dtype"],
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.n_kv_groups = n_kv_groups
        self.rope_base = rope_base
        self.dtype = dtype
        super().__init__(**kwargs)


class Embedding(torch.nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim,
        )
        self.pos_emb = torch.nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.emb_dim,
        )
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_size)
        return: (batch_size, seq_size, emb_dim)
        """
        _, seq_size = input_ids.size()
        position_ids = torch.arange(seq_size).unsqueeze(0)
        pos_emb = self.pos_emb(position_ids)
        tok_emb = self.tok_emb(input_ids)
        emb = pos_emb + tok_emb
        return emb
    
def scaled_dot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    # query: (batch_size, seq_size, head_dim)
    _, seq_size, head_dim = key.size()
    scores = torch.bmm(query, key.transpose(1, 2)) # (batch_size, seq_size, seq_size)
    scores = scores.masked_fill(mask.bool()[:seq_size, :seq_size], value=-torch.inf)
    weights = torch.nn.functional.softmax(scores / head_dim ** 0.5, dim=-1)
    return torch.bmm(weights, value)
    

class AttentionHead(torch.nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        head_dim = config.emb_dim // config.n_heads
        self.query = torch.nn.Linear(config.emb_dim, head_dim)
        self.key = torch.nn.Linear(config.emb_dim, head_dim)
        self.value = torch.nn.Linear(config.emb_dim, head_dim)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, emb_dim)
        return: (batch_sie, seq_size, emb_dim)
        """
        return scaled_dot_attention(
            self.query(x),
            self.key(x),
            self.value(x),
            self.mask,
        )
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            AttentionHead(config) for _ in range(config.n_heads)
        ])
        self.output = torch.nn.Linear(config.emb_dim, config.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, emb_dim)
        return: (batch_sie, seq_size, emb_dim)
        """
        hidden = torch.concat([
            h(x) for h in self.heads
        ], dim=-1)
        return self.output(hidden)

    

class FNN(torch.nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim, 4 * config.emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, emb_dim)
        return: (batch_sie, seq_size, emb_dim)
        """
        return self.layers(x)
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        # multiheadattention
        self.attn = MultiHeadAttention(config)
        # FNN
        self.fnn = FNN(config)
        self.norm = torch.nn.LayerNorm(config.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, emb_dim)
        return: (batch_sie, seq_size, emb_dim)
        """
        skip = x 
        x = self.norm(x)
        x = self.attn(x)
        x = skip + x 

        skip = x 
        x = self.norm(x)
        x = self.fnn(x)
        x = skip + x
        return x

class Qwen3(PreTrainedModel):
    config_class = Qwen3Config

    def __init__(self, config):
        super().__init__(config)
        # embedding
        # transformer blocks
        # output projection
        self.embedding = Embedding(config)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.output_layer = torch.nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, labels = None) -> Dict[str, torch.Tensor]:
        """
        input_ids: (batch_size, seq_size)
        labels: (batch_size, seq_size)
        """
        x = self.embedding(input_ids)
        for b in self.transformer_blocks:
            x = b(x)
        logits = self.output_layer(x)  # (batch_size, seq_size, vocab_size)
        if labels is not None:
            # TODO: flatten the losses
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten(0, 1)
            )
            return {
                "loss": loss,
                "logits": logits,
            }
        return {
            "logits": logits
        }
    

def text_to_tokens(text: str, tokenizer) -> torch.Tensor:
    return tokenizer.encode(text, padding=True, return_tensors="pt")

def tokens_to_text(tokens: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(tokens.squeeze(0))
    
def greedy_complete(prefix: str, model, tokenizer, config: Qwen3Config, max_length: int = 50) -> str:
    input_ids = text_to_tokens(prefix, tokenizer)  # (1, seq_size)
    for _ in range(max_length):
        in_context_tokens = input_ids[-config.context_length:]
        logits = model(in_context_tokens)["logits"] # (1, seq_size, vocab_size)
        token_id = torch.argmax(logits[:, -1, :], dim=-1) 

        input_ids = torch.concat([input_ids, token_id.unsqueeze(0)], dim=-1)
        if token_id == tokenizer.eos_token:
            break
    
    return tokens_to_text(input_ids, tokenizer)
    

def main():
    # load data
    with open("/Users/lkhamsurenl/development/transformers/gpt2/t8.shakespeare.txt") as f:
        text = f.read()

    train_id: int = int(0.9 * len(text))
    train_text = text[:train_id]
    test_text = text[train_id:]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config()

    train_ds = QwenDataset(train_text, tokenizer, config.context_length)
    test_ds = QwenDataset(test_text, tokenizer, config.context_length)

    model = Qwen3(config)
    text = greedy_complete(
        "Hello World!",
        model,
        tokenizer,
        config
    )
    print(f"pre-training: {text}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        "./model",
        max_steps=1,
        use_cpu=True,
        push_to_hub=False
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )
    trainer.train()
    text = greedy_complete(
        "Hello World!",
        model,
        tokenizer,
        config
    )
    print(f"post-training: {text}")


    
if __name__ == "__main__":
    main()
