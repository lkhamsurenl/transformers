from transformers import AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset

from encoder import AttentionClassifier


def tokenize_dataset_to_dataloader(data, tokenizer, shuffle: bool = False, batch_size: int = 32):
    # 1. tokenize
    dataset = data.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
        batched=True
    )
    # convert to pytorch
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    # convert to dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_one_epoch(training_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs = data["input_ids"]
        labels = data["label"].float()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 0:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

        # TODO: Move it out
        return last_loss

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # load imdb dataset
    imdb = load_dataset("imdb")

    # Create data loaders for our datasets; shuffle for imdb["train"], not for validation
    training_loader = tokenize_dataset_to_dataloader(imdb["train"], tokenizer, shuffle=True)
    test_loader = tokenize_dataset_to_dataloader(imdb["test"], tokenizer, shuffle=False)

    model = AttentionClassifier(config)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # train model
    for _ in range(3):
        avg_loss = train_one_epoch(training_loader, optimizer, model, loss_fn)

        running_tloss = 0.
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                test_inputs = test_data["input_ids"]
                test_labels = test_data["label"].float()
                test_outputs = model(test_inputs)
                test_loss = loss_fn(test_outputs, test_labels)
                running_tloss += test_loss

        avg_tloss = running_tloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_tloss))


if __name__ == '__main__':
    main()

