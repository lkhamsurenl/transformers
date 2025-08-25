import urllib.request
import os
import torch
from typing import List, Dict

from transformers import TrainingArguments, AutoTokenizer, DataCollatorWithPadding, Trainer

from model import GPT2, GPTConfig

filename = "t8.shakespeare.txt"

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, tokenizer, context_length: int, stride_length: int) -> None:
        self.input_ids = []
        self.label_ids = []
        tokens = tokenizer(text)["input_ids"]
        for i in range(0, len(tokens) - context_length, stride_length):
            input_chunk = tokens[i: i + context_length]
            label_chunk = tokens[i + 1: i + context_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.label_ids.append(torch.tensor(label_chunk))

    def __getitem__(self, idx) -> Dict:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.label_ids[idx],
        }

    def __len__(self):
        return len(self.input_ids)


def download() -> None:
    if not os.path.exists(filename):
        print(f"Downloading file into {filename}")
        urllib.request.urlretrieve(os.path.join("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/", filename), filename)
    else:
        print(f"File {filename} already exists, skipping!")


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text(prompt: str, model: GPT2, tokenizer, n_ctx: int, max_length: int = 10) -> str:
    # generate text up to max_length using model
    input_ids = text_to_token_ids(prompt, tokenizer)
    for _ in range(max_length):
        # get latest context
        context_token_ids = input_ids[-n_ctx:]
        preds = model(context_token_ids)  # (1, seq_size, vocab_size)
        new_token_id = torch.argmax(preds["logits"][:, -1, :], dim=-1)

        # TODO: Figure out how to avoid unsqueeze & squeeze
        input_ids = torch.concat((input_ids.squeeze(0), new_token_id), dim=0).unsqueeze(0)
    return token_ids_to_text(input_ids, tokenizer)

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    download()
    # get training data
    with open(filename, "r") as f:
        text = f.read()

    print(text[:100])
    train_idx = int(len(text) * 0.9)
    train_text = text[:train_idx]
    test_text = text[train_idx:]
    print(f"train_text = {train_text[:10]}; test_text = {test_text[:10]}")

    config = GPTConfig()
    model = GPT2(config)
    print(f"model = {model}")
    sample_completion = generate_text("Hello world!", model, tokenizer, config.n_ctx)
    print(f"sample_completion = {sample_completion}")

    # Create train & test dataset
    train_ds = GPTDataset(train_text, tokenizer, context_length=config.n_ctx, stride_length=config.n_ctx)
    test_ds = GPTDataset(test_text, tokenizer, context_length=config.n_ctx, stride_length=config.n_ctx)

    print(f"train = {train_ds[0]}")

    training_args = TrainingArguments(
        "test-trainer",
        push_to_hub=False,
        use_cpu=True,
        max_steps=1,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    sample_completion = generate_text("Hello world!", model, tokenizer, config.n_ctx)
    print(f"sample_completion after training = {sample_completion}")

if __name__ == "__main__":
    main()