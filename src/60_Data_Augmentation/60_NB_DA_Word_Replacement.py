# %% [markdown]
# ## Data Augmentation: Word Replacement

# %%

import os
import sys

# Needed to import modules from parent directory
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import re
import math
import time
import random
import hashlib
import statistics
from datasets import Dataset
from utils.model import *
from utils.multithreading import *
from utils.accelerators import *
from utils.preprocessing import *
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from huggingface_hub import InferenceClient
from datasets import load_from_disk, Dataset, ClassLabel, Value, Features
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from bson import ObjectId
from tqdm import tqdm
from utils.files import *
from utils.database import *
from sklearn.model_selection import train_test_split
import os
import sys

# Needed to import modules from parent directory
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# %%
topic = "cannabis"  # "energie" #"kinder" "cannabis"

# %% [markdown]
# ## Get Predictions

# %% [markdown]
# ### Load Model

# %%
MODEL_NAME = "deepset/gbert-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).eval()

# %%
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# %% [markdown]
# ### Load Dataset

# %%
# dataset = load_from_disk(f"../../data/tmp/processed_dataset_buff_{topic}_split_chunkified")
dataset = load_from_disk(
    f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_random")

dataset

# %% [markdown]
# ## Generate new Training Examples

# %% [markdown]
# ### Test on an Example

# %%


def randomly_replace_tokens(text, tokenizer, model, mask_probability=0.15):
    """ Randomly mask input tokens and predict the missing ones with a model. """

    # Tokenize the input text and prepare it for the model: Convert the text to input IDs,
    # generate attention masks (to ignore padding in the attention mechanism), and ensure
    # all inputs are of the same length by padding shorter texts and truncating longer ones.
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding='max_length', add_special_tokens=True)
    input_ids = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    replaced_input_ids = input_ids.clone()

    # Generate a random array of the same shape as input_ids. This will be used to decide
    # which tokens to mask based on the mask_probability. Tokens corresponding to 'True' in
    # this array will be considered for masking.
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < mask_probability) * (input_ids != tokenizer.cls_token_id) * \
               (input_ids != tokenizer.sep_token_id) * \
        (input_ids != tokenizer.pad_token_id)

    # Replace selected tokens with the mask token ID in input_ids.
    selection = mask_arr.nonzero(as_tuple=False)[:, 1].tolist()
    input_ids[0, selection] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    predictions = outputs.logits

    print("Selection: ", selection)
    # For each token position that was masked, find the token with the highest score (most likely token)
    # and replace the masked token with it in replaced_input_ids.
    for i in selection:
        # Get all predictions for this token and sort predictions by likelihood
        all_predictions = predictions[0, i]
        sorted_predictions = torch.argsort(all_predictions, descending=True)

        for pred_i in sorted_predictions:
            # If the predicted token is different from the original, use it

            print("Pred_i: ", pred_i)
            print("Replaced_input_ids: ", replaced_input_ids[0, i])
            if pred_i != replaced_input_ids[0, i]:
                replaced_input_ids[0, i] = pred_i
                break  # Exit the loop once a different token is found

    # Decode the replaced tokens back to a string, skipping special tokens
    replaced_text = tokenizer.decode(
        replaced_input_ids[0], skip_special_tokens=True)
    return replaced_text

# %%


def randomly_replace_tokens(text, tokenizer, model, mask_probability=0.15):
    """Elegantly replace tokens one by one, each with full context."""

    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding='max_length', add_special_tokens=True)
    input_ids = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask

    # Identify non-special tokens for potential masking
    non_special_token_indices = [i for i, token_id in enumerate(input_ids[0])
                                 if token_id not in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)]

    # Randomly select tokens for masking
    num_tokens_to_mask = int(len(non_special_token_indices) * mask_probability)
    tokens_to_mask = np.random.choice(
        non_special_token_indices, size=num_tokens_to_mask, replace=False)

    for i in tokens_to_mask:
        # Save the original token ID
        original_token_id = input_ids[0, i].item()
        masked_input_ids = input_ids.detach().clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id  # Mask the token

        with torch.no_grad():
            outputs = model(masked_input_ids, attention_mask=attention_mask)

        predictions = outputs.logits[0, i]
        predictions[original_token_id] = - \
            float('Inf')  # Invalidate the original token
        best_pred_idx = predictions.argmax(dim=-1).item()
        input_ids[0, i] = best_pred_idx  # Replace with the best prediction

    replaced_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return replaced_text


# %%
# Example usage
text = "Das hier ist ein Test."
replaced_text = randomly_replace_tokens(
    text, tokenizer, model, mask_probability=0.35)
print("Original text:", text)
print("Replaced text:", replaced_text)

# %% [markdown]
# ## Iterate over Training Dataset

# %%

# %%
# Filter positive examples and sample 20 percent of the positive exampl
positive_examples = dataset['train'].filter(
    lambda example: example['label'] == 1)

# Select the first 20% of the shuffled positive examples as your random sample
positive_examples_shuffled = positive_examples.shuffle(seed=42)
num_samples = int(len(positive_examples_shuffled) * 0.20)
sampled_examples = positive_examples_shuffled.select(range(num_samples))

# Generate new data points for the sampled positive examples
dataset[f'positive_sampled'] = sampled_examples
dataset

# %%


def generate_new_data_points(text, n_examples=1):
    """Generates n new data points from the original text."""
    new_texts = [randomly_replace_tokens(
        text, tokenizer, model, 0.35) for _ in range(n_examples)]
    return new_texts


# %%
for n in [1, 2, 3]:
    print(f"Generating {n} new examples for each original example...")

    # Placeholder for the expanded dataset
    expanded_examples = []

    # Iterate over each example in the sampled examples to generate new data points
    for example in tqdm(sampled_examples):
        new_texts = generate_new_data_points(example['text'], n)
        for new_text in new_texts:
            new_example = example.copy()
            new_example['text'] = new_text
            expanded_examples.append(new_example)

    # Convert the list of new examples to a Dataset
    expanded_dataset = Dataset.from_pandas(pd.DataFrame(expanded_examples))
    dataset[f'expanded_{n}'] = expanded_dataset

    print(f"Completed generating {n} new examples for each original example.")


# %% [markdown]
# ## Save Generated Trainig Examples

# %%
dataset

# %%
# Save the expanded dataset
dataset.save_to_disk(
    f"../../data/tmp/augmented_dataset_{topic}_word_replacement")

# %%
