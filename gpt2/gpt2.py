import torch 
from torch.utils.data import Dataset
from typing import List
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoTokenizer, 
    AutoConfig, 
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel
)
import evaluate
import numpy as np

base_model = "gpt2"
base_config = AutoConfig.from_pretrained(base_model)


class GPTDataset(Dataset):
    """Custom dataset based on Shakespeare works"""
    def __init__(self, text: str, tokenizer, context_length: int, stride_length: int):
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        token_ids: List[str] = tokenizer.encode(text)
        for i in range(0, len(token_ids), stride_length):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + 1 + context_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx], 
            "labels": self.target_ids[idx],
        }
    

class GPTConfig(PretrainedConfig):
    model_type = "gpt2"

    def __init__(
        self,
        n_ctx: int = base_config.n_ctx,
        n_embd: int = base_config.n_embd,
        n_head: int = base_config.n_head,
        vocab_size: int = base_config.vocab_size,
        n_positions: int = base_config.n_positions,
        n_layer: int = base_config.n_layer,
        num_kv_groups: int = 2, 
        **kwargs
    ):
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.num_kv_groups = num_kv_groups
        super().__init__(**kwargs)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim_size: int, epsilon: float = 1e-8):
        super().__init__()
        self.dim_size = dim_size
        self.gamma = torch.nn.Parameter(torch.empty(self.dim_size))
        self.eps = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute RMS of the x
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms
        return normed * self.gamma



class Embedding(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=config.n_positions,
            embedding_dim=config.n_embd,
        )
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
        )
        self.dropout = torch.nn.Dropout()

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (batch_size, seq_size)
        return: (batch_size, seq_size, n_embd)
        """
        _, seq_size = input_ids.size()
        position_ids = torch.arange(seq_size).unsqueeze(0) # (1, seq_size)
        pos_emb = self.position_embedding(position_ids)
        tok_emb = self.token_embedding(input_ids)
        emb = pos_emb + tok_emb
        emb = self.dropout(emb)
        return emb
    

def scaled_dot_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    query: (batch_size, seq_size, head_dim)
    mask: (n_ctx, n_ctx)
    return: (batch_size, seq_size, head_dim)
    """
    batch_size, seq_size, head_dim = key.size()
    scores = torch.bmm(query, key.transpose(1, 2)) # (batch_size, seq_size, seq_size)
    # mask the scores to ensure it's causal
    scores = scores.masked_fill(attention_mask.bool()[:seq_size, :seq_size], value=-torch.inf)
    weights = torch.softmax(scores / head_dim ** 0.5, dim=-1) # (batch_size, seq_size, seq_size)
    return torch.bmm(weights, value) # (batch_size, seq_size, head_dim)
    

class AttentionHead(torch.nn.Module):
    def __init__(self, n_embd: int, head_dim: int, n_ctx: int):
        super().__init__()
        self.query = torch.nn.Linear(n_embd, head_dim)
        self.key = torch.nn.Linear(n_embd, head_dim)
        self.value = torch.nn.Linear(n_embd, head_dim)
        self.register_buffer(
            "attention_mask", torch.triu(torch.ones(n_ctx, n_ctx), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, n_embd)
        return: (batch_size, seq_size, n_embd)
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
            AttentionHead(
                n_embd=config.n_embd,
                head_dim=head_dim,
                n_ctx=config.n_ctx,
            ) for _ in range(config.n_head)
        ])
        # TODO: potentially add dropout here
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, n_embd)
        return: (batch_size, seq_size, n_embd)
        """
        x = torch.concat([h(x) for h in self.heads], dim=-1)
        return x
    
class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.num_kv_groups = config.num_kv_groups
        self.group_size = config.n_head // config.num_kv_groups

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.n_ctx, config.n_ctx), diagonal=1)
        )

        self.queries = torch.nn.Linear(config.n_embd, self.n_head * self.head_dim)
        self.keys = torch.nn.Linear(config.n_embd, config.num_kv_groups * self.head_dim)
        self.values = torch.nn.Linear(config.n_embd, config.num_kv_groups * self.head_dim)

        self.queries_norm = RMSNorm(self.head_dim)
        self.keys_norm = RMSNorm(self.head_dim)

        self.output_proj = torch.nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, n_embd)
        return: (batch_size, seq_size, n_embd)
        """
        batch_size, seq_size, n_embd = x.size()

        queries = self.queries(x)  # (batch_size, seq_size, n_head * head_dim)
        keys = self.keys(x)  # (batch_size, seq_size, num_kv_groups * head_dim)
        values = self.values(x)  # (batch_size, seq_size, num_kv_groups * head_dim)

        queries = queries.view(batch_size, seq_size, self.n_head, self.head_dim)
        keys = keys.view(batch_size, seq_size, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, seq_size, self.num_kv_groups, self.head_dim)

        queries = queries.transpose(1, 2)  # (batch_size, n_head, seq_size, head_dim)
        keys = keys.transpose(1, 2)  # (batch_size, num_kv_groups, seq_size, head_dim)
        values = values.transpose(1, 2)  # (batch_size, num_kv_groups, seq_size, head_dim)

        queries = self.queries_norm(queries)
        keys = self.keys_norm(keys)

        keys = keys.repeat_interleave(self.group_size, dim = 1)  # (batch_size, n_head, seq_size, head_dim)
        values = values.repeat_interleave(self.group_size, dim = 1)

        scores = torch.matmul(queries, keys.transpose(2, 3))  # (batch_size, n_head, seq_size, seq_size)
        scores = scores.masked_fill(self.mask.bool()[:seq_size, :seq_size], -torch.inf)
        weights = torch.nn.functional.softmax(scores / self.head_dim ** 0.5, dim = -1)

        context = torch.matmul(weights, values)  # (batch_size, n_head, seq_size, head_dim)
        context = context.transpose(1, 2)  # (batch_size, seq_size, n_head, head_dim)
        context = context.resize(batch_size, seq_size, n_embd)
        return self.output_proj(context)
    

class FNN(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = torch.nn.Linear(config.n_embd, config.n_embd)
        self.fc2 = torch.nn.Linear(config.n_embd, config.n_embd)
        self.fc3 = torch.nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_size, n_embd)
        return: (batch_size, seq_size, n_embd)
        """
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = torch.nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)



class TransformerBlock(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.fnn = FNN(config)
        self.norm = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, seq_size, n_embd)
        return: (batch_size, seq_size, n_embd)
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
        

class GPT(PreTrainedModel):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        # TODO: This should be just layers, as opposed to one by one
        self.embedding = Embedding(config)
        self.transformers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.output = torch.nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, labels = None, attention_mask = None):
        x = self.embedding(input_ids)
        for t in self.transformers:
            x = t(x)
        logits = self.output(x) # (batch_size, seq_size, vocab_size)

        if labels is not None:
            # NOTE: This is important for to flatten the output before feeding into the cross entropy function
            loss = torch.nn.functional.cross_entropy(input=logits.flatten(0, 1), target=labels.flatten(0, 1))
            return {
                "loss": loss,
                "logits": logits
            }
        return {
            "logits": logits
        }
    

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def token_ids_to_text(input_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(input_ids.squeeze(0).tolist())


def greedy_completion(
    model,
    tokenizer,
    text: str,
    n_ctx: int,
    max_tokens_to_generate: int = 50
):
    # do completion using the model
    # tokenize
    input_ids: torch.Tensor = torch.tensor(tokenizer(text)["input_ids"]) # (1, seq_size)
    for _ in range(max_tokens_to_generate):
        context_token_ids = input_ids[-n_ctx:]
        preds = model(context_token_ids.unsqueeze(0))  # (1, seq_size, vocab_size)
        new_token_id = torch.argmax(preds["logits"][:, -1, :], dim=-1) # (1, 1)
        input_ids = torch.concat([input_ids, new_token_id])
    return token_ids_to_text(input_ids, tokenizer)


def main():
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # read data
    with open("t8.shakespeare.txt") as f:
        text = f.read()
    train_idx = int(len(text) * 0.9)
    train_text = text[:train_idx]
    test_text = text[train_idx:]

    # create train and test datasets
    train_ds = GPTDataset(
        text=train_text,
        tokenizer=tokenizer,
        context_length=base_config.n_ctx,
        stride_length=base_config.n_ctx,
    )
    test_ds = GPTDataset(
        text=test_text[:1024*8],
        tokenizer=tokenizer,
        context_length=base_config.n_ctx,
        stride_length=base_config.n_ctx,
    )
    # model
    config = GPTConfig()
    model = GPT(config)

    print(f"""
    pre-training completion:
    {greedy_completion(
            model=model,
            tokenizer=tokenizer,
            text="Hello world!",
            n_ctx=config.n_ctx
        )}
    """)
    # train model
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="./model",
        max_steps=1,
        use_cpu=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds, # TODO: Increase the data size
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # trainer.evaluate()


    print(f"""
    post-training completion:
    {greedy_completion(
            model=model,
            tokenizer=tokenizer,
            text="Hello world!",
            n_ctx=config.n_ctx
        )}
    """)
    # save model

if __name__ == '__main__':
    main()