# %% [markdown]
# ## Train Multiple LIB Models

# %%
from collections import Counter
import json
from tabulate import tabulate
from IPython.display import display, HTML
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.feature_selection import mutual_info_classif
from collections import defaultdict, Counter
from urllib.parse import urlparse, urlunparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk, concatenate_datasets
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
FEATURES = "url"  # "url", "content", "url_and_content"

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
# ## Define Model

# %%

# %%
################################# Extract n-grams from a URL #################################


def extract_ngrams(url, n):
    """Extracts n-grams of length n from the given URL."""
    return [url[i:i+n] for i in range(len(url) - n + 1)]

################################# Calculate frequencies of n-grams #################################


def calculate_ngram_frequencies(urls, genres, n_range):
    """Calculates the frequency of n-grams for each genre from the given URLs and genres."""

    ngram_freq = defaultdict(lambda: defaultdict(int))
    genre_counts = Counter(genres)

    for url, genre in zip(urls, genres):
        for n in n_range:
            ngrams = extract_ngrams(url, n)
            for ngram in ngrams:
                ngram_freq[genre][ngram] += 1

    return ngram_freq, genre_counts

################################# Calculate the mutual information for n-grams #################################


def calculate_mutual_information(ngram_freq, genre_counts):
    """Calculates mutual information for n-grams using their frequencies and genre counts."""

    ngrams = list(
        {ngram for genre in ngram_freq for ngram in ngram_freq[genre]})
    X = np.zeros((len(ngram_freq), len(ngrams)))
    y = list(genre_counts.keys())

    for i, genre in enumerate(y):
        for j, ngram in enumerate(ngrams):
            X[i, j] = ngram_freq[genre][ngram]

    mi = mutual_info_classif(X, y, discrete_features=True)
    return dict(zip(ngrams, mi))

################################# Estimate the interpolation coefficients #################################


def estimate_interpolation_coefficients(ngram_mi, n_range):
    """Estimates interpolation coefficients based on mutual information of n-grams."""

    lambda_sum = sum(ngram_mi.values())
    lambdas = {n: 0 for n in n_range}

    for ngram, mi in ngram_mi.items():
        n = len(ngram)
        lambdas[n] += mi / lambda_sum

    # Normalize the coefficients
    total = sum(lambdas.values())
    for n in lambdas:
        lambdas[n] /= total

    return lambdas

################################# Main training function #################################


def train_genre_classifier(urls, genres, n_range=(2, 3, 4)):
    """
    Trains the genre classifier by calculating n-gram frequencies, mutual information,
    and interpolation coefficients from the given URLs and genres.
    """
    # Calculate n-gram frequencies and genre counts
    ngram_freq, genre_counts = calculate_ngram_frequencies(
        urls, genres, n_range)

    # Calculate mutual information for n-grams
    ngram_mi = calculate_mutual_information(ngram_freq, genre_counts)

    # Estimate interpolation coefficients
    lambdas = estimate_interpolation_coefficients(ngram_mi, n_range)

    return ngram_freq, genre_counts, lambdas

# %%
################################# Normalize the probabilities #################################


def normalize_probs(probs):
    """Normalizes the probability dictionary so that the probabilities sum to 1."""
    total = sum(probs.values())
    if total == 0:
        # Return zero probabilities if total is zero
        return {k: 0 for k in probs}
    return {k: v / total for k, v in probs.items()}

################################# Calculate the probability of an n-gram for a given genre #################################


def calculate_ngram_prob(ngram, genre, ngram_freq, genre_counts, lambdas):
    """
    Calculates the probability of an n-gram given a genre using pre-trained frequencies and interpolation coefficients.
    """
    n = len(ngram)
    lambda_n = lambdas.get(n, 0)
    freq = ngram_freq[genre].get(ngram, 0)
    total_ngrams = sum(ngram_freq[genre].values())
    if total_ngrams == 0:
        return 1e-10  # Return a small non-zero value if no n-grams are found
    prob = (lambda_n * (freq / total_ngrams))
    return prob

################################# Extract n-grams from a URL #################################


def extract_ngrams(url, n):
    """Extracts n-grams of length n from the given URL."""
    return [url[i:i+n] for i in range(len(url) - n + 1)]

################################# Prediction function #################################


def predict_genre(url, ngram_freq, genre_counts, lambdas, n_range=(2, 3, 4)):
    """
    Predicts the genre of a given URL using pre-trained n-gram frequencies, genre counts, and interpolation coefficients.
    """
    probs = {genre: np.log(count / sum(genre_counts.values())) if count >
             0 else -np.inf for genre, count in genre_counts.items()}
    Q = []
    grams = set()

    for n in n_range:
        Q.extend(extract_ngrams(url, n))

    while Q:
        gram = Q.pop(0)
        if any(gram in ngram_freq[genre] for genre in genre_counts):
            grams.add(gram)
        else:
            if len(gram) > min(n_range):
                Q.append(gram[:-1])

    if grams:
        for genre in genre_counts:
            for gram in grams:
                ngram_prob = calculate_ngram_prob(
                    gram, genre, ngram_freq, genre_counts, lambdas)
                if ngram_prob > 0:
                    probs[genre] += np.log(ngram_prob)

    probs = normalize_probs(probs)
    predicted_genre = max(probs, key=probs.get)

    return predicted_genre, probs


# %% [markdown]
# ## Train Model

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

# %% [markdown]
# ## Evaluate Models

# %%


def get_predictions(tokenized_datasets, features, ngram_freq, genre_counts, lambdas, n_range, split="test"):
    """Use the trained model to make predictions on the test set."""

    preds = []
    labels = []
    input_data = prepare_input_data(tokenized_datasets[split], features)

    for i, input in enumerate(tqdm(input_data)):
        predicted_class, _ = predict_genre(
            input, ngram_freq, genre_counts, lambdas, n_range)
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
            f"../../data/tmp/processed_dataset_{topic}_buffed_{SAMPLING}{SUFFIX}")

        if SPLIT == "holdout":
            dataset["holdout"] = concatenate_datasets(
                [dataset["holdout"], dataset["test"]])
        # Extract the path from the URL
        dataset = dataset.map(extract_url_path, num_proc=8)
    else:
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}")

        if SPLIT == "holdout":
            dataset["holdout"] = concatenate_datasets(
                [dataset["holdout"], dataset["test"]])

        # Extract the path from the URL
        dataset = dataset.map(extract_url_path, num_proc=8)

    # Train the model
    X_train = prepare_input_data(dataset['train'], FEATURES)
    y_labels = dataset["train"]["label"]
    n_range = (2, 3, 4)
    ngram_freq, genre_counts, lambdas = train_genre_classifier(
        X_train, y_labels, n_range)
    print(f"Trained model for {topic}")

    # Use the trained model to make predictions on the test set
    preds, labels = get_predictions(
        dataset, FEATURES, ngram_freq, genre_counts, lambdas, n_range, split=SPLIT)
    metrics = calc_metrics(labels, preds)
    print(f"Metrics for {topic}: {metrics}")

    # Add answers to the dataset
    dataset[SPLIT] = dataset[SPLIT].add_column("preds", preds)
    dataset.save_to_disk(
        f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}_s_LIB_{FEATURES}_{SPLIT}")

    # Update the eval_results dictionary
    eval_results[topic] = metrics

# %%
print(eval_results)

# %% [markdown]
# ### Save Chunk Level Predictions and Output Results

# %%

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_lib_{FEATURES}_{SPLIT}_chunks.json"

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
        f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}_s_LIB_{FEATURES}_{SPLIT}")

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
file_path = f"eval_results_lib_{FEATURES}_{SPLIT}_pages.json"

# %%
# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results_pages, file)

# %%
with open(file_path, "r") as file:
    eval_results_pages = json.load(file)

# %%
