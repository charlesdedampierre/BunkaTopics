import sys

sys.path.append("../")

import os

from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()
model_path = os.getenv("MODEL_PATH")

llm = Llama(model_path=model_path)

labels = [
    "Positive",
    "Negative",
]
labels_all = ", ".join(labels)
sentence = ["every body died"]

query = f"Which of those labels represent best this sentence: {sentence}/n the labels are the following: {labels_all}. Only write the label:"
output = llm(query, max_tokens=50, echo=False)
res = output["choices"][0]["text"]


print(res)
