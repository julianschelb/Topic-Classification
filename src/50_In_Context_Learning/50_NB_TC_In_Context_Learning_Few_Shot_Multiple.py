# %% [markdown]
# ## Classifier: In Context Learing

# %%
from IPython.display import display, HTML
import json
from tabulate import tabulate
import gc
from collections import defaultdict
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex
from random import sample
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import numpy as np
import os
import time
import random
import torch

# %%
# # Limit visibility to only GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Set the device to GPU 0 if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Define Prompt Template:

# %%
SAMPLING = "random"  # "random", "stratified", "clustered", "shared_domain"
SUFFIX = "_extended"  # "", "_holdout", "_extended"
SPLIT = "test"  # "train", "test", "holdout", "extende
MAX_CONTENT_LENGTH = 384  # 496, 192
OVERLAP = 64
FEATURES = "url_and_content"  # "url", "content", "url_and_content"

# %%
DEMONSTR_SAMPLING = "knn"  # , "random_balanced", "knn", "expert"
K = 4

# %%
TOPICS = ["cannabis", "energie", "kinder"]
MODELS = [
    {
        "name": "aya-101",
        "model": "CohereForAI/aya-101",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModelForSeq2SeqLM"
    },
    {
        "name": "vicuna-13b",
        "model": "lmsys/vicuna-13b-v1.5",
        "tokenizer_class": "LlamaTokenizer",
        "model_class": "LlamaForCausalLM"
    },
    {
        "name": "vicuna-7b",
        "model": "lmsys/vicuna-7b-v1.5",
        "tokenizer_class": "LlamaTokenizer",
        "model_class": "LlamaForCausalLM"
    },
    # {
    #     "name": "FLAN-t5-base",
    #     "model": "google/flan-t5-base",
    #     "tokenizer_class": "AutoTokenizer",
    #     "model_class": "AutoModelForSeq2SeqLM"
    # },
    # {
    #     "name": "FLAN-t5-large",
    #     "model": "google/flan-t5-large",
    #     "tokenizer_class": "AutoTokenizer",
    #     "model_class": "AutoModelForSeq2SeqLM"
    # },
    # {
    #     "name": "FLAN-t5-xxl",
    #     "model": "google/flan-t5-xxl",
    #     "tokenizer_class": "AutoTokenizer",
    #     "model_class": "AutoModelForSeq2SeqLM"
    # },
    # {
    #     "name": "leo-hessianai-13b",
    #     "model": "LeoLM/leo-hessianai-13b",
    #     "tokenizer_class": "AutoTokenizer",
    #     "model_class": "AutoModelForCausalLM"
    # },
    # {
    #     "name": "leo-hessianai-7b",
    #     "model": "LeoLM/leo-hessianai-7b",
    #     "tokenizer_class": "AutoTokenizer",
    #     "model_class": "AutoModelForCausalLM"
    # },

]

# %% [markdown]
# ## Define Prompt Template:

# %% [markdown]
# **Topic Desciptions:**

# %%
topic_desciptions = {
    "kinder": {
        "name": "Child Support Act",
        "description": "The Kindergrundsicherung (basic child support) policy aims to combat child poverty by providing a fixed amount, income-dependent supplement, and educational benefits.",
        "keywords": ['kinder', 'kindergr', 'paus', 'familie', 'bundestag.de', 'arbeitsagentur.de', 'kindergrundsicherung',  'kindergeld', 'kindersicherung', 'kinderzuschlag', 'gesetz']
    },
    "cannabis": {
        "name": "Cannabis Control Act)",
        "description": "The CanG 2023 (Cannabisgesetz, Cannabis Control Act) will legalize the private cultivation of cannabis by adults for personal use and collective non-commercial cultivation",
        "keywords": ['cannabis', 'canabis', 'cannabic', 'gras', 'cbd', 'droge', 'hanf', 'thc', 'canbe', 'legal', 'legalisierung', 'gesetz', 'verein', 'entkriminali']
    },
    "energie":  {
        "name": "Renewable Energy Sources Act",
        "description": "The EEG 2023 (Erneuerbare-Energien-Gesetz, Renewable Energy Sources Act) aims to increase the share of renewable energies in gross electricity consumption to at least 80% by 2030",
        "keywords": ['energie', 'eeg', 'grün', 'gruen', 'habeck', 'climate', 'strom', 'Waerme', 'wende', 'frderung', 'förderung', 'windkraft', 'windrad', 'photovoltaik',
                     'photovoltaic', 'solar', 'heizung', 'heiz', 'gesetz', 'erneuer', 'geothermie', 'pv', 'geg']
    }
}

# %% [markdown]
# **Prompt Templates:**

# %%
############################### ZERO SHOT ###############################

PROMPT_TEMPLATE_ZERO_SHOT = """Classify the following webpage text in {lang} as topic releated or unrelated. Does it contain information about '{topic}'? Please answer with 'Yes' or 'No' only.

Topic description: {topic_desc}
Topic keywords: {topic_keyw}

URL: '''{url}'''
Text: '''{webpage_text}'''
Answer:"""

############################### FEW SHOT ###############################

PROMPT_TEMPLATE_FEW_SHOT = """Classify the following webpage text in {lang} as topic releated or unrelated. Does it contain information about '{topic}'? Please answer with 'Yes' or 'No' only.

Topic description: {topic_desc}
Topic keywords: {topic_keyw}

Examples:
{examples}

Webpage:
URL: '''{url}'''
Text: '''{webpage_text}'''
Answer:"""

############################### DEMONSTRATOR ###############################

PROMPT_TEMPLATE_EXAMPLES = """
URL: '''{url}'''
Text: '''{text}'''
Answer: '{label}'
"""

# %% [markdown]
# **Test prompt templates:**

# %%

# Example text
example_list = [
    {
        "text": "Cannabis ist eine Droge.",
        "view_url": "google.de",
        "label": "Yes"
    },
    {
        "text": "Katzen sind Tiere.",
        "view_url": "example.com",
        "label": "No"
    }
]

example_prompt_list = [PROMPT_TEMPLATE_EXAMPLES.format(
    text=example["text"], url=example["view_url"],  label=example["label"]) for example in example_list]

# print("Example prompt list:")
# print("\n".join(example_prompt_list))

topic_desciption = topic_desciptions["cannabis"]
topic_name = topic_desciption.get("name")
topic_desc = topic_desciption.get("description")
topic_keyw = topic_desciption.get("keywords")


# Test the template with a dummy text
prompt_test = PROMPT_TEMPLATE_FEW_SHOT.format(topic=topic_name, lang='German', url="example.com",  topic_desc=topic_desc, topic_keyw=", ".join(
    topic_keyw),  webpage_text='Lorem ipsum dolor sit amet, consectetur adipiscing elit.', examples="".join(example_prompt_list))
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
          # 'num_beams': 5,
          # 'num_return_sequences': 5,
          'max_new_tokens': 128,
          'min_new_tokens': 1,
          # 'output_scores': True,
          # 'repetition_penalty': 1.0,
          # 'max_length': 8192,
          'temperature': 0.3,
          'top_k': 50,
          'top_p': 0.95
          }

# %% [markdown]
# ## Helper Functions

# %%


def compile_prompt(article, template, template_example, topic, topic_desciptions, examples, lang='German'):
    """ Compiles the prompt for the given article and model."""

    # Get the topic description and keywords
    topic_desciption = topic_desciptions[topic]
    topic_name = topic_desciption.get("name")
    topic_desc = topic_desciption.get("description")
    topic_keyw = topic_desciption.get("keywords")

    # Get the text of the article
    article_text = article.get("text")
    article_lang = article.get("lang")
    article_url = article.get("view_url")
    example_prompts = [template_example.format(
        text=example["text"], url=example["view_url"], label="Yes" if example["label"] == 1 else "No") for example in examples]
    prompt = template.format(topic=topic_name, url=article_url,  topic_desc=topic_desc, topic_keyw=topic_keyw,
                             lang=article_lang, webpage_text=article_text, examples="".join(example_prompts))

    return prompt


# %%
article = {"view_url": "test.com", "text": "Lorem Ipsum", "lang": "de"}
exmaple_prompt = compile_prompt(article, PROMPT_TEMPLATE_FEW_SHOT, PROMPT_TEMPLATE_EXAMPLES,
                                "cannabis", topic_desciptions, example_list, lang='German')
print(exmaple_prompt)

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
    return 1 if "yes" in text else 0 if "no" in text else 0

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

# %%


def load_model_and_tokenizer(model_details):
    """
    Loads a model and its corresponding tokenizer based on the provided model details.
    """
    model_name = model_details['model']
    tokenizer_class = model_details['tokenizer_class']
    model_class = model_details['model_class']

    # Cohere models and FLAN models
    if tokenizer_class == "AutoTokenizer" and model_class == "AutoModelForSeq2SeqLM":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True)

    # Vicuna models
    elif tokenizer_class == "LlamaTokenizer" and model_class == "LlamaForCausalLM":
        from transformers import LlamaTokenizer, LlamaForCausalLM
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True)

    #  LeoLM models
    elif tokenizer_class == "AutoTokenizer" and model_class == "AutoModelForCausalLM":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True, trust_remote_code=False)

    else:
        raise ValueError("Model class not supported.")

    return tokenizer, model


# %% [markdown]
# **Different Sample Methods**

# %%


################### Random Sampling ############################

def sample_examples_random(dataset, k=2):
    """Samples k pairs of examples completely at random."""
    dataset_sampled = dataset.shuffle().select(range(k))
    return [example for example in dataset_sampled]


################### Random Sampling balanced ###################

def sample_examples_random_balanced(dataset, k=2):
    """Samples k pairs of examples, each pair containing one positive and one negative example."""
    # Separate the dataset into positive and negative examples
    positive_examples = [
        example for example in dataset if example['label'] == 1]
    negative_examples = [
        example for example in dataset if example['label'] == 0]

    # Sample k examples from each subset
    sampled_positive = sample(positive_examples, k)
    sampled_negative = sample(negative_examples, k)

    # Alternate between positive and negative examples to create pairs
    examples = []
    for idx in range(k):
        if idx % 2 == 0:
            examples.append(sampled_positive[idx])
        else:
            examples.append(sampled_negative[idx])

    return examples


################### KNN Sampling ############################

def sample_examples_knn(model, index, query, dataset, k=2):
    inferred_vector = model.encode(
        query, convert_to_tensor=True, show_progress_bar=False)
    sims = index.get_nns_by_vector(
        inferred_vector, k, search_k=-1, include_distances=False)
    return [dataset[idx] for idx in sims]

################### Expert Sampling ############################

# def sample_from_expert(curated_examples, topic, k=2):

#     curated_examples_topic = curated_examples[topic]
#     sampled_positive = sample(curated_examples_topic['positive_examples'], k)
#     sampled_negative = sample(curated_examples_topic['negative_examples'], k)

#     # Alternate between positive and negative examples to create pairs
#     examples = []
#     for idx in range(k):
#         if idx % 2 == 0:
#             examples.append(sampled_positive[idx])
#         else:
#             examples.append(sampled_negative[idx])

#     return examples


# %% [markdown]
# ## Load Encoder for KNN Sampling

# %%

# %%
# Load the transformer-based model
encoder = SentenceTransformer(
    'paraphrase-multilingual-mpnet-base-v2', device='cuda:0')

# %%
# Function to encode texts to embeddings


def encode_to_embedding(example):
    example['embeddings'] = encoder.encode(example['text'])
    return example


# %%


def majority_voting(answers):
    """Apply majority voting to a list of arbitrary classification answers."""
    count = Counter(answers)
    most_common = count.most_common(2)  # Get the two most common answers

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "Tie"
    return most_common[0][0]


# Example usage with arbitrary labels
answers = ["Apple", "Banana", "Apple", "Orange"]
print(f"The majority vote is: {majority_voting(answers)}")

# %% [markdown]
# **Get samples:**

# %%
# k = 2
# examples = sample_examples_random(dataset['train'], k)
# examples = sample_examples_random_balanced(dataset['train'], k)
# examples = sample_examples_knn(dataset['train'], k)
# examples = sample_from_expert(curated_examples, topic, k)
# sample_examples_knn(model, article_index, dataset["train"][0]["text"], dataset["train"], k)
# print("Examples: ", examples)

# %% [markdown]
# ## Generate Answers

# %%

# %%
eval_results = defaultdict(dict)

for model_details in MODELS:  # -------------------------------------------------------------

    # Load model
    model_name = model_details['model']
    print(f"Loading model {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_details)

    for topic in TOPICS:  # ---------------------------------------------------------------

        # Load dataset
        print(f"Loading dataset for {topic}")
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_buffed_chunkified_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}")
        # dataset['test'] = dataset['test'].filter(lambda x: x['token_count'] > 200)
        # dataset[SPLIT] = sample_random_from_dataset(dataset, n=10, subset=SPLIT)
        # print(dataset['test'][0])

        # Few-Shot ------------------------------------------------------------------------

        # Random Sampling
        if DEMONSTR_SAMPLING == "random":
            examples = sample_examples_random(dataset['train'], k=K)
            # print("Examples: ", len(examples))

        # Random Sampling (Balanced by classes)
        elif DEMONSTR_SAMPLING == "random_balanced":
            examples = sample_examples_random_balanced(dataset['train'], k=K)

        # KNN Cluster-based sampling
        elif DEMONSTR_SAMPLING == "knn":
            article_index = AnnoyIndex(
                encoder.get_sentence_embedding_dimension(), "angular")
            article_index.load(f'../../data/indices/page_index_{topic}.ann')

        # # Use curated examples
        # elif DEMONSTR_SAMPLING == "expert":
        #     examples = sample_from_expert(curated_examples, topic, k)

        # Zero-Shot ------------------------------------------------------------------------

        elif K == 0:
            examples = []
        else:
            examples = []

        # Iterate over pages in test split
        answers = []
        # ---------------------------------------------------
        for row in tqdm(dataset[SPLIT]):

            # Dynamic Sampling
            if DEMONSTR_SAMPLING == "knn" and K > 0:
                examples = sample_examples_knn(
                    encoder, article_index, row["text"], dataset["train"], K)

            # Generate answers
            prompt_template = PROMPT_TEMPLATE_FEW_SHOT if K > 0 else PROMPT_TEMPLATE_ZERO_SHOT
            prompt = compile_prompt(
                row, prompt_template, PROMPT_TEMPLATE_EXAMPLES, topic, topic_desciptions, examples)
            answers.append(generate_answers(model, tokenizer, prompt, params))

        # Add answers to the dataset
        dataset[SPLIT] = dataset[SPLIT].add_column("answers", answers)
        dataset.save_to_disk(
            f"../../data/tmp/processed_dataset_{topic}_answers_{K}s_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}_s_{model_name.split('/')[1]}_{FEATURES}_{SPLIT}_{DEMONSTR_SAMPLING}")

        # Calculate metrics
        answers_after_voting = [majority_voting(ans) for ans in answers]
        print(answers_after_voting)
        answers_parsed = [parse_response(ans) for ans in answers_after_voting]
        metrics = calc_metrics(dataset['test']['label'], answers_parsed)
        eval_results[model_name][topic] = metrics
        print(f"Metrics for {model_name}: {metrics}")

    # Clear GPU memory to avoid memory errors
    model.cpu()
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()  # Explicitly invoking garbage collection
    torch.cuda.empty_cache()  # Clear cache again after garbage collection
    time.sleep(5)

# %%
dataset['test'][0]

# %% [markdown]
# ## Save and Output Results

# %% [markdown]
# **Get chunk level predictions:**

# %%

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_icl_{K}s_{DEMONSTR_SAMPLING}_{SPLIT}_chunks.json"

# %%

# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results, file)

# %%

# Load the dictionary from the JSON file
with open(file_path, "r") as file:
    eval_results = json.load(file)

# %%
# Identify all topics (assuming all models are evaluated on the same topics)
topics = list(next(iter(eval_results.values())).keys())

# Prepare headers for the table: each topic will have four metrics
headers = ["Model"] + \
    [f"{topic} {metric}" for topic in topics for metric in [
        "Acc.", "Prec.", "Rec.", "F1"]]

# Prepare rows: one row per model, containing metrics for each topic
rows = []
for model_name_t, topics_metrics in eval_results.items():
    row = [model_name_t]  # Start with the model name
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
# **Get page level predictions:**

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
    for model_details in MODELS:  # -------------------------------------------------------------

        model_name = model_details['model']

        print(
            f"\n\n###### Evaluating model {model_name} on {topic} ###### \n\n")
        dataset = load_from_disk(
            f"../../data/tmp/processed_dataset_{topic}_answers_{K}s_{SAMPLING}{SUFFIX}_{MAX_CONTENT_LENGTH}_s_{model_name.split('/')[1]}_{FEATURES}_{SPLIT}_{DEMONSTR_SAMPLING}")

        # print(dataset)

        # Group dataset examples by URL, with a fallback to domain
        grouped_dataset = {}
        for example in tqdm(dataset[SPLIT]):
            url = example.get("view_url") or example.get("domain")
            example_filtered = {k: example[k] for k in [
                "text", "domain", "answers", "label", "category", "annotation_type", "lang"]}
            grouped_dataset.setdefault(url, []).append(example_filtered)

        # Extract labels
        labels = []
        for url, chunks in grouped_dataset.items():
            label = [chunk["label"] for chunk in chunks]
            labels.append(max(label))

        # Merge chunk level predictions
        predictions = []
        for url, chunks in grouped_dataset.items():

            for chunk in chunks:
                chunk["pred"] = majority_voting(chunk["answers"])
                chunk["pred"] = 1 if chunk["pred"] == "Yes" else 0

            preds = [chunk["pred"] for chunk in chunks]
            pred = majority_voting(
                [pred for pred in preds if pred > 0]) if max(preds) > 0 else 0
            predictions.append(pred)

        # Use the trained model to make predictions on the test set
        print(predictions)
        print(labels)
        metrics = calc_metrics(labels, predictions)
        print(f"Metrics for {model_name} on {topic}: {metrics}")

        # Update the eval_results dictionary
        eval_results_pages[model_name][topic] = metrics


# %%
grouped_dataset.keys()

# %%
# Define the file path to save the dictionary
file_path = f"eval_results_icl_{K}s_{DEMONSTR_SAMPLING}_{SPLIT}_pages.json"

# %%
# Save the dictionary to disk as JSON
with open(file_path, "w") as file:
    json.dump(eval_results_pages, file)

# %%
with open(file_path, "r") as file:
    eval_results_pages = json.load(file)

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
