## Simple classifier using transformers library

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate 
import numpy as np
from datasets import load_dataset

combined_metrics_names = evaluate.combine(["accuracy", "precision", "recall"])

# base model to fine tune
model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    return combined_metrics_names.compute(predictions=preds, references=labels)
    

dataset = (
    dataset
    .filter(lambda ex: ex["text"] is not None)
    .map(tokenize_fn, batched=True)
)

args = TrainingArguments(
    './model',
    use_cpu=True,
    push_to_hub=False,
    eval_strategy="epoch",
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(100)),
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
trainer.evaluate()
