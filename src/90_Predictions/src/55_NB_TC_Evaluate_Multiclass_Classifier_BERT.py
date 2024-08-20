# %% [markdown]
# ## Train Multiclass Classifier: BERT

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
file_path = f"../data/tmp/processed_dataset_multiclass_chunkified_{MAX_CONTENT_LENGTH}"
dataset = load_from_disk(file_path)

# dataset = dataset["valid"]

# %% [markdown]
# ## Load Model

# %%
# "../models/bert_multiclass_model_buff"
# "../models/bert_multiclass_model_buff_incr_neg"
model_path = "../models/../models/bert_multiclass_model_buff_filtered"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()

# %%
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %% [markdown]
# ## Prepare Dataset

# %%
# Tokenize the text


def tokenize_function(examples):
    return tokenizer(examples["url_path"], examples["text"], padding="max_length", truncation=True)


dataset = dataset.map(tokenize_function, batched=True)

# %% [markdown]
# ## Get Predictions

# %%


def predict_batch(batch):
    """ Perform prediction on a batch of samples in a multiclass setting. """

    # Ensure input tensors are on the correct device
    input_ids = torch.tensor(batch['input_ids']).to(device)
    attention_mask = torch.tensor(batch['attention_mask']).to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Extract probabilities for all classes and predicted classes
    # Move results back to CPU and convert to list
    batch['probas'] = predictions.cpu().tolist()
    batch['preds'] = torch.argmax(predictions, dim=-1).cpu().tolist()

    return batch


# %%
dataset = dataset.map(predict_batch, batched=True, batch_size=512)

# %%
# id = 100
# print(dataset[id]["preds"])
# print(dataset[id]["probas"])

# %%
dataset.save_to_disk(file_path + "_preds")

# %%
# # Assuming labels and preds are lists or arrays containing the true labels and predicted labels respectively
# accuracy = accuracy_score(labels, preds)
# precision_per_class = precision_score(labels, preds, average=None)
# recall_per_class = recall_score(labels, preds, average=None)
# f1_per_class = f1_score(labels, preds, average=None)

# print("Overall Accuracy: {:.2f}%".format(accuracy * 100))
# print("Precision per class: {}".format(np.round(precision_per_class, 2)))
# print("Recall per class: {}".format(np.round(recall_per_class, 2)))
# print("F1 Score per class: {}".format(np.round(f1_per_class, 2)))

# %%
