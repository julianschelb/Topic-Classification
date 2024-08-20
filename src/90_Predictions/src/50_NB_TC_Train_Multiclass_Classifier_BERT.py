# %% [markdown]
# ## Train Multiclass Classifier: BERT

# %%
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score
from collections import Counter
import random
import numpy as np
import torch
import os

# %%
topics = ["cannabis", "energie", "kinder"]

# %% [markdown]
# ## Load Dataset

# %% [markdown]
# **Map class-names to class-ids:**

# %%
id_to_class = {0: "other", 1: "cannabis", 2: "energie", 3: "kinder"}
class_to_id = {"other": 0, "cannabis": 1, "energie": 2, "kinder": 3}

# %%
MAX_CONTENT_LENGTH = 384
dataset = load_from_disk(
    f"../data/tmp/processed_dataset_multiclass_chunkified_{MAX_CONTENT_LENGTH}_filtered")

# %%
dataset

# %% [markdown]
# ## Prepare Data

# %%
MODEL_NAME = "FacebookAI/xlm-roberta-base"  # "FacebookAI/xlm-roberta-large"

# %%
# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Tokenize the text


def tokenize_function(examples):
    return tokenizer(examples["url_path"], examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %% [markdown]
# ## Prepare Model

# %%
# Load a pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4)

# %% [markdown]
# ## Train Model

# %%


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# %%
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,             # number of training epochs
    # Weight decay if we apply some form of weight regularization.
    weight_decay=0.01,
    logging_dir='./logs',  # Directory where the training logs will be stored.
    logging_strategy="steps",  # The logging strategy determines when to log
    logging_steps=100,  # Number of steps between logging of training loss.
    evaluation_strategy="steps",  # Evaluation is done
    eval_steps=100,  # Number of steps between evaluations.
    load_best_model_at_end=True,  # load the best model at the end of training.
    metric_for_best_model="eval_loss",
    lr_scheduler_type='linear',  # The scheduler type to use, e.g., 'linear', 'cosine'
    # Proportion of training to perform linear learning rate warmup for.
    warmup_ratio=0.1
)

# %%

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics  # ,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# %%
# Train the model
trainer.train()

# %% [markdown]
# ## Evaluate Model

# %%
tokenized_datasets

# %%
# Evaluate the model
trainer.evaluate(tokenized_datasets["valid"])

# %% [markdown]
# ## Save Model

# %%
local_path = "../models/bert_multiclass_model_buff_filtered"
trainer.save_model(local_path)
tokenizer.save_pretrained(local_path)
