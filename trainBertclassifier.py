import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np

# Load data
df = pd.read_csv("descriptions.csv").dropna(subset=["Description", "Category"])
df = df.sample(frac=1, random_state=42)  # shuffle

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Category"])

# Train-test split (stratify to keep balanced classes)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Description"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    stratify=df["label"],
    random_state=42,
)

# Create Hugging Face datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pretrained model with classification head
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Training args
training_args = TrainingArguments(
    output_dir="./bert_expense_classifier",
    evaluation_strategy="epoch",   # evaluate once per epoch
    save_strategy="epoch",          # save checkpoint once per epoch (must match evaluation_strategy)
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model and label encoder
model.save_pretrained("./bert_expense_classifier")
tokenizer.save_pretrained("./bert_expense_classifier")

import joblib
joblib.dump(le, "label_encoder.joblib")
print("✅ Saved model and label encoder")
