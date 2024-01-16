# getting-started

Start by importing Bunka

## Importing Bunka

```bash
pip install bunkatopics
```

```python
from bunkatopics import Bunka
```

For this example, we'll use a dataset of tweets from the HuggingFace datasets Hub

## Loading an example

```python
import random
from datasets import load_dataset

dataset = load_dataset("rguo123/trump_tweets")["train"]["content"]
docs = random.sample(dataset, 1000)
```

Bunka leverages embedding models for language understanding. Here, we choose a small model using the langchain framework

## Initialising Bunka

```python
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bunka = Bunka(language = 'english') # Bunka is multilangual but you need to chose a multilangual embedding model as well.
```

## Topic Modeling

Fit Bunka to your documents and perform initial topic modeling:

```python
bunka.fit(docs)
bunka.get_topics(n_clusters=10) # Chose a number of topics

```

## Summarizing Topics with an LLM

To make the topics clearer, use a large language model (LLM) for summarization.
Replace "your_huggingface_api_token" with your actual Hugging Face API token.

```python
from langchain.llms import HuggingFaceHub

repo_id = "mistralai/Mistral-7B-Instruct-v0.1" # Using Mistral Model from Huggingface hub
llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token="your_huggingface_api_token",
)

df_topics_clean = bunka.get_clean_topic_name(llm=llm)
print(df_topics_clean)
```

## Visualizing Topics

Finally, visualize the topics:
This function creates a plot showing the different topics extracted from your dataset.

```python
bunka.visualize_topics(width=800, height=800)
```
