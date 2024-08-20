# ===========================================================================
#                            Helpers for Preprocessing
# ===========================================================================

from multiprocessing import Pool
from datasets import Dataset
from typing import List
from tqdm import tqdm
import numpy as np
import re

# ---------------------------------  Basic Preprocessing --------------------------------


def cleanArticleText(text):
    """Replace line breaks with a spaceoc currences of more than one space"""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return re.sub(' +', ' ', text)


def cleanArticles(articles):
    """Clean text of all articles in the given list"""

    for article in tqdm(articles, total=len(articles), desc="Cleaning articles"):
        if "parsing_result" in article:
            text = article.get("parsing_result", {}).get("text", "")
            article["parsing_result"]["text"] = cleanArticleText(text)
        else:
            print("No parsing result for article", article["_id"])

    return articles

# ---------------------------------  Preprocessing --------------------------------


def splitText(text, n_tokens, tokenizer, overlap=10):
    """Splits the input text into chunks with n_tokens tokens using HuggingFace tokenizer, with an overlap of overlap tokens."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+n_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        i += n_tokens - overlap

    return chunks


def calcInputLength(tokenizer, prompt):
    """Calculate the length of the input after"""
    return tokenizer(prompt, return_tensors="pt").input_ids.real.shape[1]


def expandRow(row, tokenizer, template, col_name="text", n_tokens=450, overlap=10,  roles=['hero', 'villain', 'victim']):
    """
    Generate prompts based on various roles and text chunks from the input row.
    """
    prompts = []

    # Split the text into chunks
    text_chunks = splitText(row.get(col_name), n_tokens, tokenizer, overlap)

    # Generate prompts for each role and text chunk
    for role in roles:
        for chunk_id, text_chunk in enumerate(text_chunks):
            prompt = template.format(elt=role, article_text=text_chunk)
            new_row = {
                **row,
                'prompt': prompt,
                'role': role,
                'chunk': chunk_id,
                'chunk_length': calcInputLength(tokenizer, text_chunk)
            }
            prompts.append(new_row)

    return prompts


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


def tokenizeInputs(example, tokenizer, col_name="prompt", params={}):
    """Tokenize the inputs"""

    # Tokenize the inputs and update the example
    max_length = tokenizer.model_max_length
    tokenized_inputs = tokenizer(
        example[col_name], max_length=max_length, **params)
    example.update(tokenized_inputs)

    return example


# --------------------------------- Datasets --------------------------------


def splitDataset(dataset: Dataset, num_chunks: int = 1) -> List[Dataset]:
    """Splits a dataset into a specified number of chunks."""

    if num_chunks < 1:
        raise ValueError("num_chunks should be at least 1.")

    # Get an array of indices from 0 to len(dataset) - 1
    indices = np.arange(len(dataset))
    # Split the indices into num_chunks parts
    split_indices = np.array_split(indices, num_chunks)

    # Create chunks using the split indices
    chunks = [Dataset.from_dict(dataset[idx_list])
              for idx_list in split_indices]

    return chunks


def convertListToDataset(articles, column_names):
    """Converts a list of articles to a HuggingFace Dataset"""

    # Ensure that "_id" is always processed even if it's not in column_names
    for article in articles:
        article["_id"] = str(article.get("_id", ""))

    dataset_dict = {}
    for col in column_names:
        # e.g., "parsing_result.text" -> ["parsing_result", "text"]
        keys = col.split(".")

        # Extracting nested data
        values = []
        for article in articles:
            temp = article
            for key in keys:
                temp = temp.get(key, None)
                if temp is None:
                    break
            values.append(temp)

        # Adding column to dataset using the last key as column name
        dataset_dict[keys[-1]] = values

    # Assuming you have the Dataset class imported
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def describeDataset(dataset):
    """
    Print basic descriptive information about the given dataset.
    """

    # Basic information
    print("Number of rows:", len(dataset))
    print("Column names:", dataset.column_names)
    print("Features (schema):", dataset.features)

    # Check if dataset has a 'set' column (e.g., 'train', 'validation', 'test')
    if "set" in dataset.column_names:
        set_counts = dataset["set"].value_counts()
        for set_name, count in set_counts.items():
            print(f"Number of samples in {set_name}: {count}")
