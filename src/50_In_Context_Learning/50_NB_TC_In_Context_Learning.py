# %% [markdown]
# ## Classifier: In Context Learing

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM
from IPython.display import display, HTML
import json
from tabulate import tabulate
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from utils.model import *
import numpy as np
import os
import random
import torch

# %%
topics = ["cannabis", "energie", "kinder"]

# %% [markdown]
# ## Load Model

# %%
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-13b", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("LeoLM/leo-hessianai-13b", trust_remote_code=True, device_map="auto", load_in_8bit=True)

# %%
# Load model directly

# model_name = "CohereForAI/aya-101"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(
#    model_name, device_map="auto", load_in_8bit=True)

# %%

# "google/flan-t5-large" # "google/flan-t5-xxl"
model_name = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True)


# %%
# model_name = "lmsys/vicuna-7b-v1.5"  # "lmsys/vicuna-7b-v1.5"
#
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(
#    model_name, device_map="auto", load_in_8bit=True)

# %% [markdown]
# ## Define Prompt Template:

# %%
PROMPT_TEMPLATE = """Given the following text in {lang}, does it contain information about '{topic}'? Please answer with 'Yes' or 'No' only.

Text: "{webpage_text}"

Answer:"""

# Test the template with a dummy text
prompt_test = PROMPT_TEMPLATE.format(
    topic="Cannabis", lang='German', webpage_text='Lorem ipsum dolor sit amet, consectetur adipiscing elit.')
print(prompt_test)

# %% [markdown]
# ## Define Parameter for Text Generation

# %% [markdown]
# Each parameter influences the text generation in a specific way. Below are the parameters along with a brief explanation:
#
# **`max_length`**:
# * Sets the maximum number of tokens in the generated text (default is 50).
# * Generation stops if the maximum length is reached before the model produces an EOS token.
# * A higher `max_length` allows for longer generated texts but may increase the time and computational resources required.
#
# **`min_length`**:
# * Sets the minimum number of tokens in the generated text (default is 10).
# * Generation continues until this minimum length is reached even if an EOS token is produced.
#
# **`num_beams`**:
# * In beam search, sets the number of "beams" or hypotheses to keep at each step (default is 4).
# * A higher number of beams increases the chances of finding a good output but also increases the computational cost.
#
# **`num_return_sequences`**:
# * Specifies the number of independently computed sequences to return (default is 3).
# * When using sampling, multiple different sequences are generated independently from each other.
#
# **`early_stopping`**:
# * Stops generation if the model produces the EOS (End Of Sentence) token, even if the predefined maximum length is not reached (default is True).
# * Useful when an EOS token signifies the logical end of a text (often represented as `</s>`).
#
# **`do_sample`**:
# * Tokens are selected probabilistically based on their likelihood scores (default is True).
# * Introduces randomness into the generation process for diverse outputs.
# * The level of randomness is controlled by the 'temperature' parameter.
#
# **`temperature`**:
# * Adjusts the probability distribution used for sampling the next token (default is 0.7).
# * Higher values make the generation more random, while lower values make it more deterministic.
#
# **`top_k`**:
# * Limits the number of tokens considered for sampling at each step to the top K most likely tokens (default is 50).
# * Can make the generation process faster and more focused.
#
# **`top_p`**:
# * Also known as nucleus sampling, sets a cumulative probability threshold (default is 0.95).
# * Tokens are sampled only from the smallest set whose cumulative probability exceeds this threshold.
#
# **`repetition_penalty`**:
# * Discourages the model from repeating the same token by modifying the token's score (default is 1.5).
# * Values greater than 1.0 penalize repetitions, and values less than 1.0 encourage repetitions.
#

# %%
params = {'do_sample': True,
          'early_stopping': True,
          # 'max_length': 100,
          # 'min_length': 1,
          # 'logprobs': 1,
          # 'n': 1,
          # 'best_of': 1,

          # 'num_beam_groups': 2,
          'num_beams': 2,
          'num_return_sequences': 1,
          'max_new_tokens': 1024,
          'min_new_tokens': 1,
          'output_scores': True,
          # 'repetition_penalty': 1.0,
          'temperature': 0.6,
          'top_k': 50,
          'top_p': 1.0
          }

# %% [markdown]
# ## Helper Functions

# %%


def compile_prompt(article, template, topic, lang='German'):
    """ Compiles the prompt for the given article and model."""

    # Extract the article headline and text
    article_text = article.get("text")
    prompt = template.format(topic=topic, lang=lang, webpage_text=article_text)
    # prompt = template.format(topic = "Cannabis", lang = 'German', webpage_text=article_text, positive_example=positive_example, negative_example=negative_example)

    return prompt

# %%


def calculate_input_length(prompt):
    """ Calculates the length of the input sequence for the model. """

    # Tokenize the prompt
    tokenized_prompt = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, truncation=False, padding=False)

    # Calculate the length of the input sequence
    input_length = tokenized_prompt.input_ids.size(1)

    return input_length

# %%


def generate_answers(model, tokenizer, prompt, params, remove_input=True):
    """Generates answers from a language model for a given prompt."""

    # Encode the prompt and generate the answers
    encoded_input = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    if encoded_input.size()[1] > tokenizer.model_max_length:
        print("Input too long, truncating.")
        # encoded_input = encoded_input[:, :tokenizer.model_max_length]

    generated_outputs = model.generate(encoded_input, **params)

    # Decode and clean outputs
    outputs = []
    input_text_wo_st = tokenizer.decode(
        encoded_input[0], skip_special_tokens=True)
    for output in generated_outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        cleaned_text = decoded_text.replace(input_text_wo_st, "").strip()
        outputs.append(cleaned_text if remove_input else decoded_text)

    return outputs

# %%


def parse_response(output_text):
    """Determines if the model's output signifies "Yes" (1) or "No" (0)."""
    text = output_text.lower()
    return 1 if "yes" in text else 0 if "no" in text else ValueError("Ambiguous response.")


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
# ## Generate Answers

# %%
eval_results = defaultdict(dict)

for topic in topics:  # ----------------------------------------------------------------------

    print(f"Loading dataset for {topic}")
    dataset = load_from_disk(
        f"../data/tmp/processed_dataset_{topic}_buffed_chunkified_random")
    # dataset['test'] = sample_random_from_dataset(dataset, n=5, subset='test')

    answers = []
    # -----------------------------------------------------
    for row in tqdm(dataset['test']):
        prompt = compile_prompt(row, PROMPT_TEMPLATE, topic)
        answers.append(generate_answers(model, tokenizer, prompt, params)[0])

    # Add answers to the dataset
    dataset['test'] = dataset['test'].add_column("answers", answers)
    dataset.save_to_disk(
        f"../data/tmp/processed_dataset_{topic}_answers_{model_name.split('/')[1]}")

    # Calculate metrics
    metrics = calc_metrics(dataset['test']['label'], [
                           parse_response(ans) for ans in answers])
    eval_results[model_name][topic] = metrics


# %%
dataset['test'][0]

# %% [markdown]
# Accuracy: 0.834
# Accuracy: 0.94

# %% [markdown]
# ## Save and Output Results

# %%

# %%

# Define the file path to save the dictionary
file_path = "eval_results_icl.json"

# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results, file)


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


# %%
