## This file uses pretrained model as feature extractor to train downstream simple model for classification

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datasets import load_dataset
import numpy as np

dataset = load_dataset("dair-ai/emotion")
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

def extract_hidden_state_fn(examples):
    inputs = {k: v for k, v in examples.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        hidden_states = model(**inputs).last_hidden_state
    return {
        "hidden_state": hidden_states[:, 0].cpu().numpy()
    }

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataset = dataset.map(extract_hidden_state_fn, batched=True)

X_train = np.array(dataset["train"]["hidden_state"])
y_train = np.array(dataset["train"]["label"])
X_val = np.array(dataset["validation"]["hidden_state"])
y_val = np.array(dataset["validation"]["label"])

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_val = clf.predict(X_val)

print("Train:\n", classification_report(y_true=y_train, y_pred=pred_train))
print("Validation:\n", classification_report(y_true=y_val, y_pred=pred_val))