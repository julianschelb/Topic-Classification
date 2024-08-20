# ===========================================================================
#                            Model Wrapper Class
# ===========================================================================
# This file contains the Model class, which is a wrapper for the Huggingface
# Inference API, OpenAI API and local model.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient
import openai
import math

# ============================ Local Model ==================================


class LocalModel:
    """A wrapper class for the local Huggingface model."""

    def __init__(self, model_name="google/flan-t5-base"):
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto")
        self.max_input_length = self.tokenizer.model_max_length

    def generateAnswer(self, input_text, params={}):
        """Generate an answer based on the input_text."""

        # Generate the output using the model
        input_ids = self.tokenizer.encode(
            input_text, return_tensors="pt", max_length=self.max_input_length, truncation=True)
        output = self.model.generate(input_ids.to("cuda"), **params)
        generated_text = self.tokenizer.decode(
            output[0], skip_special_tokens=True)

        return generated_text

    def get_max_input_length(self):
        """Returns the maximum input length allowed by the model."""
        return self.max_input_length

    def calcInputLength(self, prompt):
        """Calculate the length of the input after"""
        return self.tokenizer(prompt, return_tensors="pt").input_ids.real.shape[1]

# Testing the LocalModel class
# model = LocalModel(model_name="google/flan-t5-base")
# print(model.generateAnswer("What's the meaning of life?"))
# print(model.get_max_input_length())


# ============================ Remote Model =================================


class RemoteModel:
    """A wrapper class for using an API with the OpenAI specifications."""

    def __init__(self, api_key, api_base="http://merkur72.inf.uni-konstanz.de:8080/v1", model_name="vicuna-13b-v1.3"):
        self.name = model_name
        openai.api_key = api_key
        openai.api_base = api_base
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"lmsys/{model_name}")
        self.max_input_length = self.tokenizer.model_max_length

    def generateAnswer(self, input_text, params={}):
        """Generate an answer using the remote API."""

        try:
            # Generate the output using the model
            completion = openai.Completion.create(
                model=self.model_name,
                prompt=input_text,
                **params
            )
        except Exception as e:
            print(e)
            return " "

        return completion.choices[0].text

    def generateAnswers(self, input_text, params={}):
        """Generate an answer using the remote API."""

        try:
            # Generate the output using the model
            completion = openai.Completion.create(
                model=self.model_name,
                prompt=input_text,
                **params
            )
        except Exception as e:
            print(e)
            return []

        return completion.choices

    def get_max_input_length(self):
        """Returns the maximum input length allowed by the model."""
        return self.max_input_length

    def calcInputLength(self, prompt):
        """Calculate the length of the input after"""
        return self.tokenizer(prompt, return_tensors="pt").input_ids.real.shape[1]


# Testing the RemoteModel class
# (ensure you replace "YOUR_API_KEY" with your actual key)
# model = RemoteModel(model_name="vicuna-13b-v1.5-16k", api_key="EMPTY")
# print(model.generateAnswer("Once upon a time"))
# print(model.get_max_input_length())
# print(model.calcInputLength("Once upon a time"))


# ============================ Huggingface Model ============================

class HuggingfaceModel:
    """A wrapper class for using the Inference API by Huggingface."""

    def __init__(self, api_key, model_name="meta-llama/Llama-2-70b-chat-hf"):
        self.name = model_name
        self.api_key = api_key
        self.model_name = model_name
        self.max_input_length = math.inf

    def generateAnswer(self, input_text, params={}):
        """Generate an answer using the remote API."""

        client = InferenceClient(model=self.model_name, token=self.api_key)
        output = client.text_generation(input_text, **params)

        return output

    def get_max_input_length(self):
        """Returns the maximum input length allowed by the model."""
        return self.max_input_length

    def calcInputLength(self, prompt):
        """Calculate the length of the input after"""
        return 0  # TODO: Implement this

# Testing the RemoteModel class
# (ensure you replace "YOUR_API_KEY" with your actual key)
# model = HuggingfaceModel(model_name="EleutherAI/gpt-neox-20b",#api_key="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")
# print(model.generateAnswer("Once upon a time"))
