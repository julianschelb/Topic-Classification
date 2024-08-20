# %% [markdown]
# ## Data Augmentation: Evaluate

# %%

import os
import sys

# Needed to import modules from parent directory
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import concatenate_datasets
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from itertools import cycle
from tabulate import tabulate
import json
import os
import sys

# Needed to import modules from parent directory
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


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
topic = "cannabis"

# %%
model_name = "deepset/gbert-large"

# %%
# Define training arguments
TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',            # directory for storing logs
    logging_strategy="steps",        # log training loss every specified number of steps
    logging_steps=200,                # number of steps to log training loss
    evaluation_strategy="steps",     # evaluate model after a specified number of steps
    eval_steps=200,                   # number of steps to evaluate model
)

# %% [markdown]
# ## Load Dataset

# %%
dataset = load_from_disk(
    f"../../data/tmp/augmented_dataset_{topic}_word_replacement")
dataset

# %%


# %% [markdown]
# ## Train Models

# %%
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# %%
datasets_splits = ['positive_sampled', 'expanded_1', 'expanded_2',
                   'expanded_3', "train"]  # , #'expanded_4', 'expanded_5']

negative_samples = dataset["train"].filter(lambda x: x['label'] == 0)

# %%

# %%


def balance_and_discard_classes(dataset, label_column='label'):
    """ Balance the dataset by randomly sampling the same number of examples for each class."""

    # Aggregate indices by class
    class_indices = {label: [i for i, example in enumerate(dataset) if example[label_column] == label]
                     for label in set(dataset[label_column])}

    # Determine the size of the smallest class for balancing
    min_class_size = min(len(indices) for indices in class_indices.values())

    # Randomly sample indices from each class to match the smallest class size
    balanced_indices = [index for indices in class_indices.values(
    ) for index in random.sample(indices, min_class_size)]
    random.shuffle(balanced_indices)

    # Determine discarded indices by finding the difference between all indices and the balanced ones
    all_indices = set(range(len(dataset)))
    discarded_indices = list(all_indices - set(balanced_indices))

    # Select the balanced and discarded indices to create new datasets
    balanced_dataset = dataset.select(balanced_indices)
    discarded_dataset = dataset.select(discarded_indices)

    return balanced_dataset, discarded_dataset

# %%
# balanced_dataset, discarded_dataset = balance_and_discard_classes(dataset, 'label')

# print(f"Balanced dataset size: {len(balanced_dataset)}")
# print(f"Discarded dataset size: {len(discarded_dataset)}")

# %%


def get_predictions(tokenized_datasets, tokenizer, model, device):
    """Use the trained model to make predictions on the test set."""

    preds = []
    labels = []
    for row in tokenized_datasets:
        # Encode the text inputs
        inputs = tokenizer(row["text"], return_tensors="pt",
                           padding=True, truncation=True)
        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs.to(device))
            # Apply softmax to logits to get probabilities
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Get the predicted class (the one with the highest probability)
            predicted_class = torch.argmax(predictions).item()

        # Store the predictions and labels
        preds.append(predicted_class)
        labels.append(row["label"])

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


eval_result_list = {}

for split in datasets_splits:  # ----------------------------------------------------------------------

    print(f"Loading dataset for {split}")

    ################################## COMPILE DATASET ##################################

    if split == 'positive_sampled':
        dataset_augmented = concatenate_datasets(
            [negative_samples, dataset[split]])
    elif split == 'train':
        dataset_augmented = dataset["train"]
    else:
        positive_sampled = dataset["positive_sampled"]
        generated_samples = dataset[split]
        dataset_augmented = concatenate_datasets(
            [negative_samples, positive_sampled, generated_samples])
        print("Dataset generated samples", len(generated_samples))

    ################################## Balance Classes ##################################

    dataset_augmented, _ = balance_and_discard_classes(
        dataset_augmented, 'label')
    label_counts = Counter(dataset_augmented['label'])
    print("Dataset size", len(dataset_augmented))
    print("Class frequencies:", label_counts)

    ################################## Load Model ##################################

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    ################################## Tokenize Dataset ##################################

    # Tokenize dataset
    tokenized_dataset_train = dataset_augmented.map(lambda examples:
                                                    tokenizer(
                                                        examples["text"], padding="max_length", truncation=True),
                                                    batched=True)

    tokenized_dataset_test = dataset["test"].map(lambda examples:
                                                 tokenizer(
                                                     examples["text"], padding="max_length", truncation=True),
                                                 batched=True)

    # Shuffle both datasets
    tokenized_dataset_train = tokenized_dataset_train.shuffle(seed=42)
    tokenized_dataset_test = tokenized_dataset_test.shuffle(seed=42)

    ################################## Class Weights ##################################

    # Calculate class weights inversely proportional to class frequencies
    labels = dataset_augmented['label']
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights", class_weights_tensor)

    ################################## Train Model ##################################

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Create a Trainer object
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    ################################## EVALUATE ##################################

    # Evaluate the model
    eval_results = trainer.evaluate(tokenized_dataset_test)
    print("Eval Results:", eval_results)

    # Use the trained model to make predictions on the test set
    preds, labels = get_predictions(
        tokenized_dataset_test, tokenizer, model, "cuda")
    metrics = calc_metrics(labels, preds)
    eval_result_list[split] = metrics

    ################################## Save MODEL ##################################

    # # Save the model
    # local_path = f"../models/{model_name.replace('/','_')}_{topic}_model"
    # trainer.save_model(local_path)
    # tokenizer.save_pretrained(local_path)

    ################################## CLEAR GPU ##################################

    # Clear GPU memory to avoid memory errors
    del model, tokenizer, tokenized_dataset_test, tokenized_dataset_train, trainer
    torch.cuda.empty_cache()


# %%
print(eval_result_list)

# %%

# Example dictionary with evaluation results per dataset
# results = {
#     "Dataset_A": {'accuracy': 0.9, 'precision': 0.91, 'recall': 0.92, 'f1': 0.93},
#     "Dataset_B": {'accuracy': 0.85, 'precision': 0.86, 'recall': 0.87, 'f1': 0.88},
#     "Dataset_C": {'accuracy': 0.95, 'precision': 0.96, 'recall': 0.97, 'f1': 0.98},
#     # Add more datasets as needed
# }

results = eval_result_list

metrics = list(results[next(iter(results))].keys())
datasets = list(results.keys())

# Extendable setup for plotting
# Add more colors if needed
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
# Add more markers if needed
markers = cycle(['o', '^', 's', 'd', 'p', '*', 'x'])

plt.figure(figsize=(10, 6))
for metric in metrics:
    metric_values = [results[dataset][metric] for dataset in datasets]
    plt.plot(datasets, metric_values, marker=next(markers),
             color=next(colors), label=metric.capitalize())

plt.xlabel('Dataset')
plt.ylabel('Metric Value')
plt.title('Evaluation Metrics per Dataset')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save and Output Results

# %%

# %%

# Define the file path to save the dictionary
file_path = f"eval_results_da_wr_{topic}.json"

# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results, file)


# %%

# Define the file path where the JSON data is saved
file_path = f"eval_results_da_wr_{topic}.json"

# Load the dictionary from the JSON file
with open(file_path, "r") as file:
    eval_results = json.load(file)
