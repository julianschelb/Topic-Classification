# %% [markdown]
# ## Train Multiple Classifiers

# %%
import json
from transformers import EarlyStoppingCallback
from collections import defaultdict
from urllib.parse import urlparse, urlunparse
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score
import random
import numpy as np
import torch
import os

# %% [markdown]
# ## Set Random Seed for Reproducibility

# %%
# Set a seed for random module
random.seed(777)

# Set a seed for numpy module
np.random.seed(777)

# Set a seed for torch module
torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# ## Define Parameters

# %%
RUN = "2"
TOPICS = ["cannabis", "kinder", "energie"]
MODEL = "deepset/gelectra-large"  # "deepset/gelectra-large"
STRATEGIES = ["random", "stratified", "clustered"]  # , "shared_domain"]
SUFFIX = "_extended"  # "", "_holdout", "_extended"
MAX_CONTENT_LENGTH = 384  # 496, 192, 384
OVERLAP = 64
FEATURES = "url_and_content"  # "url", "content", "url_and_content"

# %%

# %%
# Define training arguments
TRAINING_ARGS = TrainingArguments(
    output_dir=f"./results_sampling_{RUN}",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,             # number of training epochs # TODO
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

# %% [markdown]
# **Extract URL-path:**

# %%


def extract_url_path(example):
    view_url = example['view_url']
    if "://" not in view_url:
        view_url = "http://" + view_url  # Assume http if no protocol specified
    parsed_url = urlparse(view_url)
    new_url = urlunparse(
        ('', '', parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))
    example['url_path'] = new_url.lstrip(
        '/')  # Store the result in a new field
    return example


extract_url_path(
    {"view_url": "https://www.google.com/search?q=python+url+path"})

# %% [markdown]
# ## Train Models

# %%


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# %%
for topic in TOPICS:  # ----------------------------------------------------------------------
    for strategie in STRATEGIES:  # -------------------------------------------------------------
        print(f"Training on {topic} dataset using {strategie} strategy")
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{strategie}{SUFFIX}_{MAX_CONTENT_LENGTH}")

# %%

training_results = defaultdict(dict)

for topic in TOPICS:  # ----------------------------------------------------------------------

    for strategie in STRATEGIES:  # -------------------------------------------------------------

        print(f"Training on {topic} dataset using {strategie} strategy")

        if FEATURES == "url":
            dataset = load_from_disk(
                f"../../data/tmp/processed_dataset_{topic}_buffed_{strategie}")
            # Extract the path from the URL
            dataset = dataset.map(extract_url_path)
            # dataset['test'] = sample_random_from_dataset(dataset, n=5, subset='test')
        else:
            dataset = load_from_disk(
                f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{strategie}{SUFFIX}_{MAX_CONTENT_LENGTH}")
            # Extract the path from the URL
            dataset = dataset.map(extract_url_path)
            # dataset['test'] = sample_random_from_dataset(dataset, n=5, subset='test')

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=2)

        # Define tokenization strategies
        tokenization_strategies = {

            # Tokenize the content of the page
            "content": lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),

            # Tokenize the URL path
            "url": lambda examples: tokenizer(examples["url_path"], padding="max_length", truncation=True),

            # Tokenize the URL path and the content of the page
            "url_and_content": lambda examples: tokenizer(examples["url_path"], examples["text"], padding="max_length", truncation=True)
        }

        # Tokenize dataset
        tokenized_datasets = dataset.map(tokenization_strategies[FEATURES],
                                         batched=True)

        # Create a Trainer object
        trainer = Trainer(
            model=model,
            args=TRAINING_ARGS,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the model
        training_result = trainer.train()

        # Update the eval_results dictionary
        training_results[strategie][topic] = training_result

        # Evaluate the model
        eval_results = trainer.evaluate(tokenized_datasets["test"])
        print("Eval Results:", eval_results)

        # Save the model
        local_path = f"../models_ccu/{MODEL.split('/')[0]}_sampling_{strategie.replace('/','_')}_{topic}_model_{FEATURES}_run_{RUN}"
        trainer.save_model(local_path)
        tokenizer.save_pretrained(local_path)

        # Clear GPU memory to avoid memory errors
        del model, tokenizer, tokenized_datasets, trainer
        torch.cuda.empty_cache()

# %%

# Define the file path to save the dictionary
file_path = f"training_results_sampling_{FEATURES}_{RUN}.json"

# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(training_results, file)

# %%
