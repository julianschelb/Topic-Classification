# %% [markdown]
# ## Train Classifier: BERT

# %%
from transformers import EarlyStoppingCallback
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from datasets import concatenate_datasets
from multiprocessing import Pool
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sklearn.model_selection import train_test_split
from collections import Counter
import datasets
from datasets import ClassLabel, Features, Value, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score
import random
import numpy as np
import torch
import os

# %%
topics = ["cannabis", "energie", "kinder"]

# %% [markdown]
# ## Load Dataset

# %%
id_to_class = {0: "other", 1: "cannabis", 2: "energie", 3: "kinder"}
class_to_id = {"other": 0, "cannabis": 1, "energie": 2, "kinder": 3}

# %%
dataset_cannabis = load_from_disk(
    f"../../data/tmp/processed_dataset_cannabis_buffed")
dataset_energie = load_from_disk(
    f"../../data/tmp/processed_dataset_energie_buffed")
dataset_kinder = load_from_disk(
    f"../../data/tmp/processed_dataset_kinder_buffed")

# %%

# Define the new class label feature
class_labels = ClassLabel(names=["other", "cannabis", "energie", "kinder"])

# Define the new features, including all existing ones plus the updated 'label'
new_features = datasets.Features({
    '_id': datasets.Value('string'),
    'batch_id': datasets.Value('int64'),
    'domain': datasets.Value('string'),
    'view_url': datasets.Value('string'),
    'lang': datasets.Value('string'),
    'text': datasets.Value('string'),
    'topic': datasets.Value('string'),
    'category': datasets.Value('string'),
    'good_for_training': datasets.Value('string'),
    'good_for_augmentation': datasets.Value('string'),
    'annotation_type': datasets.Value('string'),
    'text_length': datasets.Value('int64'),
    'word_count': datasets.Value('int64'),
    'is_topic': datasets.Value('int64'),
    'label': class_labels  # Use the updated ClassLabel feature
})

# Function to update the dataset with new features


def update_dataset_schema(dataset):
    return dataset.cast(new_features)


# Assuming dataset_cannabis, dataset_energie, dataset_kinder are loaded and need schema updates
dataset_cannabis = update_dataset_schema(dataset_cannabis.map(lambda e: {
                                         'label': class_to_id['cannabis'] if e["label"] == 1 else class_to_id['other']}, features=new_features))

dataset_energie = update_dataset_schema(dataset_energie.map(lambda e: {
                                        'label': class_to_id['energie'] if e["label"] == 1 else class_to_id['other']}, features=new_features))

dataset_kinder = update_dataset_schema(dataset_kinder.map(lambda e: {
                                       'label': class_to_id['kinder'] if e["label"] == 1 else class_to_id['other']}, features=new_features))


# %%
# Concatenate all datasets
dataset_all_topics = concatenate_datasets(
    [dataset_cannabis, dataset_energie, dataset_kinder])

# %%
dataset_all_topics

# %%
dataset_all_topics[0]

# %%
# Filter positive and negative examples
dataset_all_topics_pos = dataset_all_topics.filter(
    lambda example: example['label'] > 0, num_proc=16)
dataset_all_topics_neg = dataset_all_topics.filter(
    lambda example: example['label'] == 0, num_proc=16)

# %%
# Collect all view_url values from dataset_all_topics_pos
pos_view_urls = set(dataset_all_topics_pos['view_url'])

# Filter dataset_all_topics_neg to exclude any rows present in dataset_all_topics_pos
dataset_all_topics_neg = dataset_all_topics_neg.filter(
    lambda example: example['view_url'] not in pos_view_urls, num_proc=16)


# %%
dataset_all_topics_neg

# %% [markdown]
# ## Deduplicate Negative Examples

# %%
seen_urls = set()

dataset_all_topics_neg = dataset_all_topics_neg.filter(
    lambda example: example['view_url'] not in seen_urls and not seen_urls.add(example['view_url']), num_proc=16)

# %% [markdown]
# ## Sample Negative Examples

# %%
print("Number of distinct domains", len(set(dataset_all_topics_neg["domain"])))

# %%
# Step 1: Calculate domain frequencies

domain_counts = Counter(dataset_all_topics_neg['domain'])

N = 512  # Keep the top 512 most frequent domains
top_domains = set([domain for domain, count in domain_counts.most_common(N)])

# Step 2: Mark all other domains as "other"


def group_domains(example):
    if example['domain'] not in top_domains:
        example['domain'] = 'other'
    return example

# Apply the transformation to the dataset


dataset_all_topics_neg = dataset_all_topics_neg.map(group_domains)

# %%

# Convert to Pandas DataFrame
df_dataset = dataset_all_topics_neg.to_pandas()

sample_size = len(dataset_all_topics_pos)
print("Positive Sample size:", sample_size)

# Perform stratified sampling
_, stratified_sample = train_test_split(
    df_dataset, test_size=sample_size, stratify=df_dataset['domain'], random_state=42)
stratified_sample.reset_index(drop=True, inplace=True)

stratified_sample.head()

# %%
dataset_all_topics_neg_sampled = Dataset.from_pandas(stratified_sample)
dataset_all_topics_neg_sampled = dataset_all_topics_neg_sampled.cast(
    new_features)

# %%
# Concatenate all datasets
dataset_all_topics = concatenate_datasets(
    [dataset_all_topics_pos, dataset_all_topics_neg_sampled])

# %%
dataset_all_topics

# %%
split_datasets = dataset_all_topics.train_test_split(
    test_size=0.1, shuffle=True)

# %%
print("Size of training set:", len(split_datasets['train']))
print("Size of testing set:", len(split_datasets['test']))

# %% [markdown]
# ## Chunkify Examples

# %%

# %%
MODEL_NAME = "deepset/gbert-large"
MAX_CONTENT_LENGTH = 512
OVERLAP = 64

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%


def get_input_length(text):
    """ Tokenize the input text and return the number of tokens """
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False, padding=False))


print(get_input_length("Hello, my name is John Doe"))

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CONTENT_LENGTH,
    chunk_overlap=OVERLAP,
    length_function=get_input_length,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)

# text_splitter = CharacterTextSplitter(
#     separator = ".", # Split text by sentences
#     chunk_size=MAX_CONTENT_LENGTH,
#     chunk_overlap=OVERLAP,
#     length_function=get_input_length,
#     is_separator_regex=False,
# )

# %%
test_text = split_datasets['train'][0]['text']
print(test_text)

# %%
texts = text_splitter.split_text(test_text)
print(len(texts))

print(texts[0])
print("Length of text:", get_input_length(texts[0]))


# %%
def expandRow(row, col_name):
    """
    Generate prompts based on text chunks from the input row.
    """
    rows = []

    # Split the text into chunks
    text_chunks = text_splitter.split_text(row.get(col_name, ""))

    # Generate prompts for each text chunk
    for chunk_id, text_chunk in enumerate(text_chunks):
        new_row = {
            **row, 'chunk_id': chunk_id, 'text': text_chunk
        }
        rows.append(new_row)

    return rows


# %%


def processDataset(dataset, num_processes, func, params=()):
    """Process a list of articles in parallel using a multiprocessing Pool."""

    # Creates a list of arguments for each call to func
    # Uses starmap to pass multiple arguments to func
    with Pool(processes=num_processes) as pool:
        args = [(row,) + params for row in dataset]
        dataset = list(pool.starmap(func, args))

    # Flatten the resulting list of lists
    # and convert it into a Dataset
    dataset = [item for sublist in dataset for item in sublist]
    dataset = Dataset.from_dict(
        {key: [dic[key] for dic in dataset] for key in dataset[0]})

    return dataset


# %%
split_datasets

# %%
num_processes = 24
params = ("text",)

for split in split_datasets:
    split_datasets[split] = processDataset(
        split_datasets[split], num_processes, expandRow, params)

# %%
split_datasets

# %%
split_datasets["train"][0]

# %%
dataset = split_datasets

# %% [markdown]
# ## Prepare Data

# %%
MODEL_NAME = "deepset/gbert-large"

# %%
# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Tokenize the text


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


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
# tokenized_datasets = tokenized_datasets["train"].shuffle(seed=42)
# tokenized_datasets["train"] = tokenized_datasets["train"].select(range(1000))


def stratified_sample(dataset, n=2000, seed=42):
    # Determine unique labels and samples per class
    labels = dataset['label']
    unique_labels = set(labels)
    samples_per_class = n // len(unique_labels)

    # Stratified sampling
    samples = [
        dataset.filter(lambda example: example['label'] == label).shuffle(
            seed=seed).select(range(min(samples_per_class, dataset.num_rows)))
        for label in unique_labels
    ]

    # Concatenate samples from all classes
    return concatenate_datasets(samples)


# Perform stratified sampling on the training dataset
# tokenized_datasets['train'] = stratified_sample(
#     tokenized_datasets['train'], n=2000, seed=42)

# tokenized_datasets


# %%

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# %%
# Define training arguments
TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,             # number of training epochs # TODO
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

# Assuming `train_dataset` is your training dataset and 'labels' are your target labels
labels = dataset["train"]['label']

# Calculate class weights inversely proportional to class frequencies
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(labels), y=labels)

# Convert class weights to a tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(class_weights_tensor)

# %%


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


# %%
# Create a Trainer


trainer = Trainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
trainer.evaluate(tokenized_datasets["test"])

# %% [markdown]
# ## Save Model

# %%
locsl_path = "../models/bert_multiclass_model_buff"
trainer.save_model(locsl_path)
tokenizer.save_pretrained(locsl_path)

# %% [markdown]
# ## Test Model

# %%
id = 1
input_ids = tokenized_datasets["test"][id]["input_ids"]
label = tokenized_datasets["test"][id]["label"]
print(label)
print(tokenizer.decode(input_ids))
# print(input_ids)


# %%
output = model(input_ids=torch.tensor([input_ids], device='cuda'))
print(output.logits)

# %%
probabilities = torch.nn.functional.softmax(output.logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1)
print(predicted_class)

# %% [markdown]
# **Constructed Positive Example**

# %%

# Your test example
test_example = """
Verständnis zur Legalisierung von Cannabis

Die Legalisierung von Cannabis, auch bekannt als Marihuana, ist ein Thema bedeutender Debatten und politischer Veränderungen in verschiedenen Ländern weltweit. Die Bewegung hin zur Legalisierung repräsentiert einen Wandel in der Wahrnehmung und Regulierung von Cannabis, von einer streng kontrollierten Substanz hin zu einer liberaler regulierten, oft sowohl für medizinische als auch für Freizeitzwecke.

Historischer Kontext: Traditionell war Cannabis in den meisten Teilen der Welt illegal, klassifiziert neben vielen anderen kontrollierten Substanzen. Diese Klassifizierung erfolgte hauptsächlich aufgrund von Bedenken hinsichtlich seines Potenzials für Missbrauch, seiner psychoaktiven Effekte und möglicher Gesundheitsrisiken.
"""

# Tokenize the text
inputs = tokenizer(test_example, return_tensors="pt",
                   padding=True, truncation=True, max_length=512)

# Predict the class
with torch.no_grad():
    outputs = model(**inputs.to('cuda'))
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted class (the one with the highest probability)
predicted_class = torch.argmax(predictions).item()

# Print the predicted class
print("Predicted class:", predicted_class)

# %% [markdown]
# **Constructed Negative Example**

# %%

# Your test example
test_example = """
Die Faszination Süßer Katzen

Katzen sind faszinierende und unglaublich beliebte Haustiere. Ihre Anmut, Unabhängigkeit und das spielerische Wesen machen sie zu einem Liebling vieler Menschen. Besonders süße Katzen haben eine besondere Anziehungskraft, die das Herz vieler Tierliebhaber erobert.

Eleganz und Anmut: Katzen sind bekannt für ihre elegante und anmutige Art. Mit ihren geschmeidigen Bewegungen und dem majestätischen Gang ziehen sie die Aufmerksamkeit auf sich. Ihre Fähigkeit, sich leise und behände zu bewegen, verleiht ihnen eine fast mystische Aura.
"""

# Tokenize the text
inputs = tokenizer(test_example, return_tensors="pt",
                   padding=True, truncation=True, max_length=512)

# Predict the class
with torch.no_grad():
    outputs = model(**inputs.to('cuda'))
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted class (the one with the highest probability)
predicted_class = torch.argmax(predictions).item()

# Print the predicted class
print("Predicted class:", predicted_class)

# %%
tokenized_datasets["test"]

# %%
# Calculate the accuracy on the test dataset
preds = []
labels = []
for row in tokenized_datasets["test"]:
    inputs = tokenizer(row["text"], return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs.to('cuda'))
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions).item()

    preds.append(predicted_class)
    labels.append(row["label"])

# %%

# Assuming labels and preds are lists or arrays containing the true labels and predicted labels respectively
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='weighted')
recall = recall_score(labels, preds, average='weighted')
f1 = f1_score(labels, preds, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))


# %%
# Assuming labels and preds are lists or arrays containing the true labels and predicted labels respectively
accuracy = accuracy_score(labels, preds)
precision_per_class = precision_score(labels, preds, average=None)
recall_per_class = recall_score(labels, preds, average=None)
f1_per_class = f1_score(labels, preds, average=None)

print("Overall Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision per class: {}".format(np.round(precision_per_class, 2)))
print("Recall per class: {}".format(np.round(recall_per_class, 2)))
print("F1 Score per class: {}".format(np.round(f1_per_class, 2)))

# %%

# Assuming labels and preds are lists or arrays containing the true labels and predicted labels respectively
cm = confusion_matrix(labels, preds)

# Create a seaborn heatmap with annotations
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=True)

# Add labels and title
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")

# Show plot
plt.show()


# %%
# Show some misclassified examples
for i in range(len(labels)):
    if labels[i] != preds[i]:
        print(tokenized_datasets["test"][i]["text"])
        print("True label:", labels[i])
        print("Predicted label:", preds[i])
        print("")

# %%
