from modules import GPTModel
import tiktoken
import torch
import numpy as np
import pandas as pd


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def autocomplete(model: torch.nn.Module, tokens: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """
    Generate text completion given input tokens

    :param model: (batch_size, seq_size, vocab_size)
    :param tokens: (batch_size, seq_size)
    :param max_new_tokens: Number of new tokens to generate
    :param context_size: size of the context to use for each generation
    :return:
    """
    for _ in range(max_new_tokens):
        current_context_tokens = tokens[:, -context_size:]
        with torch.no_grad():
            output_tokens = model(current_context_tokens)  # (batch_size, seq_size, vocab_size)
        probs = torch.softmax(output_tokens[:, -1, :], dim = -1)  # (batch_size, vocab_size)
        new_tokens = torch.argmax(probs, dim = -1, keepdim=True)
        tokens = torch.concat((tokens, new_tokens), dim = 1)  # (batch_size, seq_size + 1)
    return tokens



def main():
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello my name is"
    input_tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    # put the model in eval mode
    model.eval()
    output_tokens = autocomplete(model, input_tokens, 6, GPT_CONFIG_124M["context_length"])
    output_text = tokenizer.decode(output_tokens.squeeze(0).tolist())
    print(output_text)


if __name__ == '__main__':
    main()
