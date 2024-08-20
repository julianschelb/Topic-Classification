# %% [markdown]
# ## Evaluate Multiple Sampling Strategies

# %%
from collections import Counter
import json
from tabulate import tabulate
from IPython.display import display, HTML
from collections import defaultdict
from urllib.parse import urlparse, urlunparse
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import random
from tqdm import tqdm
import numpy as np
import torch
import os

# %% [markdown]
# ## Set Random Seed for Reproducibility

# %%
# Set a seed for random module
random.seed(42)

# Set a seed for numpy module
np.random.seed(42)

# Set a seed for torch module
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# ## Define Parameters

# %%
RUN = "1"
TOPICS = ["cannabis", "kinder", "energie"]  # ["cannabis", "kinder", "energie"]
MODEL = "deepset/gelectra-large"  # "deepset/gelectra-large"
STRATEGIES = ["random", "stratified", "clustered"]  # , "shared_domain"]
SUFFIX = "_extended"  # "", "_holdout", "_extended"
SPLIT = "extended"  # "train", "test", "holdout", "extended"
MAX_CONTENT_LENGTH = 384  # 496, 192, 384
OVERLAP = 64
FEATURES = "url_and_content"  # "url", "content", "url_and_content"


# %%

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
# ## Evaluate Models

# %%

# %%


def get_predictions(tokenized_datasets, tokenizer, model, device, features, batch_size=128, split="test"):
    """Use the trained model to make predictions on the test set."""

    preds = []
    labels = []
    probabilities = []

    # Prepare the data loader
    data_loader = DataLoader(tokenized_datasets[split], batch_size=batch_size)

    for batch in tqdm(data_loader):
        # Encode the text inputs
        if features == "content":
            inputs = tokenizer(
                batch["text"], padding="max_length", truncation=True, return_tensors="pt")
        elif features == "url":
            inputs = tokenizer(
                batch["url_path"], padding="max_length", truncation=True, return_tensors="pt")
        elif features == "url_and_content":
            inputs = tokenizer(batch["url_path"], batch["text"],
                               padding="max_length", truncation=True, return_tensors="pt")
        else:
            raise ValueError(
                "Invalid value for FEATURES. Expected 'content', 'url', or 'url_and_content'.")

        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs.to(device))
            # Apply softmax to logits to get probabilities
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Get the predicted classes (the ones with the highest probability)
            predicted_classes = torch.argmax(predictions, dim=-1).cpu().numpy()

        # Store the predictions, labels, and probabilities
        preds.extend(predicted_classes.tolist())
        labels.extend(batch["label"].tolist())
        # Store the probability of the positive class
        probabilities.extend(predictions.cpu().numpy()[:, 1].tolist())

    return preds, labels, probabilities

# %%


def calc_metrics(labels, preds):
    """
    Calculates the accuracy, precision, recall, and F1 score for the given labels and predictions and returns them in a dictionary.
    """

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary'),
        'recall': recall_score(labels, preds, average='binary'),
        'f1': f1_score(labels, preds, average='binary'),
    }

    return metrics


# %%
def sample_random_from_dataset(dataset, n=5, subset='test'):
    """
    Samples n random examples from a specified subset of the dataset.
    """
    n = min(n, len(dataset[subset]))
    random_indices = random.sample(range(len(dataset[subset])), n)
    sampled_dataset = dataset[subset].select(random_indices)
    return sampled_dataset

# %% [markdown]
# **Get chunk level predictions:**


# %%
eval_results = defaultdict(dict)

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

            if SPLIT == "holdout":
                dataset["holdout"] = concatenate_datasets(
                    [dataset["holdout"], dataset["test"]])

            # Extract the path from the URL
            dataset = dataset.map(extract_url_path)
            # dataset['test'] = sample_random_from_dataset(dataset, n=5, subset='test')

        # Load model and tokenizer
        model_name_local = f"../models_ccu/{MODEL.split('/')[0]}_sampling_{strategie.replace('/','_')}_{topic}_model_{FEATURES}_run_{RUN}"
        print(f"Loading model from {model_name_local}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_local, num_labels=2, local_files_only=True)

        # # Use multiple GPUs if available
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = torch.nn.DataParallel(model)

        # Move model to GPU if available
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(DEVICE)

        # Use the trained model to make predictions on the test set
        preds, labels, probas = get_predictions(
            dataset, tokenizer, model, DEVICE, FEATURES, split=SPLIT)
        metrics = calc_metrics(labels, preds)
        print(f"Metrics for {MODEL} on {topic}: {metrics}")

        # Add answers to the dataset
        dataset[SPLIT] = dataset[SPLIT].add_column("preds", preds)
        dataset[SPLIT] = dataset[SPLIT].add_column("probas", probas)
        dataset.save_to_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{strategie}{SUFFIX}_{MAX_CONTENT_LENGTH}_sampling_{MODEL.split('/')[1]}_{FEATURES}_{SPLIT}_run_{RUN}")

        # Update the eval_results dictionary
        eval_results[strategie][topic] = metrics

        # Clear GPU memory to avoid memory errors
        del model, tokenizer
        torch.cuda.empty_cache()


# %%
print(eval_results)

# %% [markdown]
# ### Save Chunk Level Predictions and Output Results

# %%

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_sampling_{FEATURES}_{SPLIT}_chunks_run_{RUN}.json"

# %%
# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results, file)

# with open(file_path, "r") as file:
#     eval_results = json.load(file)

# %%
# Identify all topics (assuming all models are evaluated on the same topics)
topics = list(next(iter(eval_results.values())).keys())

# Prepare headers for the table: each topic will have four metrics
headers = ["Model"] + \
    [f"{topic} {metric}" for topic in topics for metric in [
        "Acc.", "Prec.", "Rec.", "F1"]]

# Prepare rows: one row per model, containing metrics for each topic
rows = []
for model, topics_metrics in eval_results.items():
    row = [model]  # Start with the model name
    for topic in topics:
        metrics = topics_metrics.get(topic, {})
        row.extend([metrics.get('accuracy', 0.0), metrics.get(
            'precision', 0.0), metrics.get('recall', 0.0), metrics.get('f1', 0.0)])
    rows.append(row)

# Generate the HTML table
table_html = tabulate(rows, headers=headers, tablefmt="html",
                      showindex="never", floatfmt=".3f")

# %%
display(HTML(table_html))


# %% [markdown]
# ## Page Level Predictions

# %%

# %%

def majority_voting(answers):
    """Apply majority voting to a list of arbitrary classification answers."""
    count = Counter(answers)
    most_common = count.most_common()  # Get all common answers sorted by frequency

    if not most_common:
        return 0  # Handle empty input scenario

    # Check for ties at the highest count
    max_votes = most_common[0][1]
    tied_classes = [cls for cls, votes in most_common if votes == max_votes]

    if len(tied_classes) > 1:
        # Return the maximum class label in case of a tie
        return max(tied_classes)
    return tied_classes[0]  # Return the class with the most votes


majority_voting([1, 1, 2, 2, 2, 3])

# %%
eval_results_pages = defaultdict(dict)

for topic in TOPICS:  # ----------------------------------------------------------------------

    for strategie in STRATEGIES:  # -------------------------------------------------------------

        print(f"\n\n###### Evaluating model {MODEL} on {topic} ###### \n\n")
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{strategie}{SUFFIX}_{MAX_CONTENT_LENGTH}_sampling_{MODEL.split('/')[1]}_{FEATURES}_{SPLIT}_run_{RUN}")

        print(dataset)

        # Group dataset examples by URL, with a fallback to domain
        grouped_dataset = {}
        for example in tqdm(dataset[SPLIT]):
            url = example.get("view_url") or example.get("domain")
            example_filtered = {k: example[k] for k in [
                "text", "domain", "preds", "label", "category", "annotation_type", "lang"]}
            grouped_dataset.setdefault(url, []).append(example_filtered)

        # Extract labels
        labels = []
        for url, chunks in grouped_dataset.items():
            preds = [chunk["label"] for chunk in chunks]
            labels.append(max(preds))

        # Merge chunk level predictions
        predictions = []
        for url, chunks in grouped_dataset.items():
            preds = [chunk["preds"] for chunk in chunks]
            pred = majority_voting(
                [pred for pred in preds if pred > 0]) if max(preds) > 0 else 0
            predictions.append(pred)

        # Use the trained model to make predictions on the test set
        metrics = calc_metrics(labels, predictions)
        print(f"Metrics for {strategie} on {topic}: {metrics}")

        # Update the eval_results dictionary
        eval_results_pages[strategie][topic] = metrics


# %% [markdown]
# ### Save Chunk Level Predictions and Output Results

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_sampling_{FEATURES}_{SPLIT}_pages_run_{RUN}.json"

# %%
# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results_pages, file)

# with open(file_path, "r") as file:
#     eval_results_pages = json.load(file)

# %%
# Identify all topics (assuming all models are evaluated on the same topics)
topics = list(next(iter(eval_results_pages.values())).keys())

# Prepare headers for the table: each topic will have four metrics
headers = ["Model"] + \
    [f"{topic} {metric}" for topic in topics for metric in [
        "Acc.", "Prec.", "Rec.", "F1"]]

# Prepare rows: one row per model, containing metrics for each topic
rows = []
for model, topics_metrics in eval_results_pages.items():
    row = [model]  # Start with the model name
    for topic in topics:
        metrics = topics_metrics.get(topic, {})
        row.extend([metrics.get('accuracy', 0.0), metrics.get(
            'precision', 0.0), metrics.get('recall', 0.0), metrics.get('f1', 0.0)])
    rows.append(row)

# Generate the HTML table
table_html = tabulate(rows, headers=headers, tablefmt="html",
                      showindex="never", floatfmt=".3f")

# %%
display(HTML(table_html))

# %%
