import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from modules import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "embed_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer, context_length: int, stride: int):
        """
        Create new Dataset object from input data.

        :param text: Full input data in text format
        :param tokenizer: tokenizer used for encoding
        :param context_length: maximum length of the sequence
        :param stride: stride used for each sequence
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - context_length, stride):
            # sequence forming next context_length tokens are input
            input_chunk = token_ids[i: i + context_length]
            # sequence that's shifted by 1 with context_length is target
            target_chunk = token_ids[i + 1: i + context_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    tokenizer,
    text: str,
    batch_size: int,
    context_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 0
):
    dataset = GPTDatasetV1(text, tokenizer, context_length=context_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def text_to_token_ids(tokenizer, text: str) -> torch.Tensor:
    token_ids = tokenizer.encode(text)
    return torch.tensor(token_ids).unsqueeze(0)


def token_ids_to_text(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def autocomplete(model: torch.nn.Module, tokens: torch.Tensor, max_new_tokens: int, context_length: int) -> torch.Tensor:
    """
    Generate text completion given input tokens

    :param model: (batch_size, seq_size, vocab_size)
    :param tokens: (batch_size, seq_size)
    :param max_new_tokens: Number of new tokens to generate
    :param context_length: size of the context to use for each generation
    :return:
    """
    for _ in range(max_new_tokens):
        current_context_tokens = tokens[:, -context_length:]
        with torch.no_grad():
            output_tokens = model(current_context_tokens)  # (batch_size, seq_size, vocab_size)
        probs = torch.softmax(output_tokens[:, -1, :], dim=-1)  # (batch_size, vocab_size)
        new_tokens = torch.argmax(probs, dim=-1, keepdim=True)
        tokens = torch.concat((tokens, new_tokens), dim=1)  # (batch_size, seq_size + 1)
    return tokens


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        # NOTE: This flattening is necessary because cross entropy expects input of
        # logits = (batch_size * seq_size, vocab_size)
        # targets = (batch_size * seq_size)
        # Cross entropy loss is used for multi-class tasks. targets are label token's index.
        logits.flatten(0, 1), target_batch.flatten(0, 1)
    )
    return loss


def calc_loss_loader(dataloader: DataLoader, model: torch.nn.Module, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def evaluate_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device, eval_iter: int):
    # Easier to do evaluation on train and val dataset at the same time using eval() & no_grad()
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: torch.nn.Module, tokenizer, device, start_context: str) -> None:
    """
    Generate autocompletion based on the start_context using the current model

    :param model:
    :param tokenizer:
    :param device:
    :param start_context:
    :return:
    """
    model.eval()
    input_tokens = text_to_token_ids(tokenizer, start_context).to(device)
    with torch.no_grad():
        token_ids = autocomplete(model, input_tokens, max_new_tokens=50, context_length=GPT_CONFIG_124M["context_length"])
    generated_text = token_ids_to_text(tokenizer, token_ids)
    print(f"Generated sample: {generated_text}")
    model.train()


def train_model_simple(
    model: torch.nn.Module,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer,
    device,
    eval_freq: int,
    eval_iter: int,
    start_context: str
):
    """
    Perform training on a given pytorch model.

    :param model: Model to train
    :param tokenizer: tokenizer used
    :param train_loader:
    :param val_loader:
    :param num_epochs: Number of epochs to train the model
    :param optimizer: optimizer used for training (e.g. AdamW)
    :param device: cuda or cpu
    :param eval_freq: Frequency of evaluation w.r.t global_step
    :param eval_iter: How many batches to use for evaluation. If None, use all data available
    :param start_context: token_ids to generate sample output to test the model improvement
    :return:
    """
    train_losses, val_losses = [], []
    global_step = -1

    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                # evaluate the model
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step}): "
                      f"train loss: {train_loss}, "
                      f"val loss: {val_loss}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses


def main():
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    file_path = "/Users/lkhamsurenl/development/transformers/notebooks/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[:split_idx]

    train_loader = create_dataloader_v1(
        tokenizer,
        train_data,
        batch_size=2,
        context_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        tokenizer,
        val_data,
        batch_size=2,
        context_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model_simple(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        optimizer=optimizer,
        device=device,
        eval_freq=5,
        eval_iter=5,
        start_context="Hello world"
    )


if __name__ == '__main__':
    main()
