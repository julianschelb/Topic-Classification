# %% [markdown]
# ## Train Multiple Random Forrest Models

# %%
from collections import Counter
import json
from tabulate import tabulate
from IPython.display import display, HTML
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from urllib.parse import urlparse, urlunparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
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
SAMPLING = "random"  # "random", "stratified", "clustered", "shared_domain"
SUFFIX = "_extended"  # "", "_holdout", "_extended"
SPLIT = "extended"  # "train", "test", "holdout", "extende
MAX_CONTENT_LENGTH = 384  # 496, 192
OVERLAP = 64
FEATURES = "content"  # "url", "content", "url_and_content"

# %%
TOPICS = ["cannabis", "kinder", "energie"]
# TOPICS = ["cannabis"]

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
# ## Train Model

# %%
nltk.download('stopwords')

german_stop_words = stopwords.words('german')

# %%

# %%


def prepare_input_data(train_data, features):
    """Prepares input data based on the specified features."""
    X_train = []
    for row in train_data:
        if features == "content":
            input = row["text"]
        elif features == "url":
            input = row["url_path"]
        elif features == "url_and_content":
            input = row["url_path"] + " " + row["text"]
        else:
            raise ValueError(
                "Invalid value for features. Expected 'content', 'url', or 'url_and_content'.")
        X_train.append(input)
    return X_train


def train_model_rf(train_data, german_stop_words, features, max_features=10000):
    """Trains an RF model and returns the model and vectorizer."""

    X_train = prepare_input_data(train_data, features)

    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer(
        stop_words=german_stop_words, max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)
    y_train = np.array(train_data['label'])

    # Train an SVM classifier
    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier, vectorizer


def get_predictions_rf(model, vectorizer, new_data):
    """Gets predictions for new data using the trained model and vectorizer."""
    X_new = vectorizer.transform(new_data)
    return model.predict(X_new)

# %% [markdown]
# ## Evaluate Models

# %%


def get_predictions(tokenized_datasets, model, vectorizer, features, split="test"):
    """Use the trained model to make predictions on the test set."""

    preds = []
    labels = []
    input_data = prepare_input_data(tokenized_datasets[split], features)

    for i, input in enumerate(tqdm(input_data)):
        predicted_class = get_predictions_rf(model, vectorizer, [input])[0]
        preds.append(predicted_class)
        labels.append(tokenized_datasets[split][i]["label"])

    return preds, labels

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

    print(f"\n\n###### Evaluating model on {topic} ###### \n\n")

    if FEATURES == "url":
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_{SAMPLING}")
        # Extract the path from the URL
        dataset = dataset.map(extract_url_path, num_proc=8)
    else:
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}")
        # Extract the path from the URL
        dataset = dataset.map(extract_url_path, num_proc=8)

    # Train Model
    model, vectorizer = train_model_rf(
        dataset['train'], german_stop_words, FEATURES)

    # Use the trained model to make predictions on the test set
    preds, labels = get_predictions(
        dataset, model, vectorizer, FEATURES, split=SPLIT)
    metrics = calc_metrics(labels, preds)
    print(f"Metrics for {topic}: {metrics}")

    # Add answers to the dataset
    dataset[SPLIT] = dataset[SPLIT].add_column("preds", preds)
    dataset.save_to_disk(
        f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}_s_RF_{FEATURES}_{SPLIT}")

    # Update the eval_results dictionary
    eval_results[topic] = metrics

    # Clear GPU memory to avoid memory errors
    del model, vectorizer

# %%
print(eval_results)

# %% [markdown]
# ### Save Chunk Level Predictions and Output Results

# %%

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_rf_{FEATURES}_{SPLIT}_chunks.json"

# %%
# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results, file)

# %%
with open(file_path, "r") as file:
    eval_results = json.load(file)

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

    print(f"\n\n###### Evaluating on {topic} ###### \n\n")
    dataset = load_from_disk(
        f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}_s_RF_{FEATURES}_{SPLIT}")

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
    print(f"Metrics for {topic}: {metrics}")

    # Update the eval_results dictionary
    eval_results_pages[topic] = metrics


# %%
print(eval_results_pages)

# %% [markdown]
# ### Save Chunk Level Predictions and Output Results

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_rf_{FEATURES}_{SPLIT}_pages.json"

# %%
# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results_pages, file)

# %%
with open(file_path, "r") as file:
    eval_results_pages = json.load(file)

# %%
