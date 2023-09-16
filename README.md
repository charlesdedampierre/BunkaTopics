[![PyPI - Python](https://img.shields.io/badge/python-v3.9+-blue.svg)](https://pypi.org/project/bertopic/)
[![PyPI - PyPi](https://img.shields.io/pypi/v/bunkatopics)](https://pypi.org/project/bunkatopics/)
[![Downloads](https://static.pepy.tech/badge/bunkatopics)](https://pepy.tech/project/bunkatopics)
[![Downloads](https://static.pepy.tech/badge/bunkatopics/month)](https://pepy.tech/project/bunkatopics)

# Bunkatopics

<img src="images/logo.png" width="35%" height="35%" align="right" />

Bunkatopics is a Topic Modeling Visualisation Method that leverages Transformers from HuggingFace through langchain. It is built with the same philosophy as [BERTopic](https://github.com/MaartenGr/BERTopic) but goes deeper in the visualization to help users grasp quickly and intuitively the content of thousands of text.
It also allows for a supervised visual representation by letting the user create continnums with natural language.

## Installation

First, create a new virtual environment using pyenv

```bash
pyenv virtualenv 3.9 bunkatopics_env
```

Activate the environment

```bash
pyenv activate bunkatopics_env
```

Then Install the Bunkatopics package:

```bash
pip install bunkatopics
```

Install the spacy tokenizer model for english:

```bash
python -m spacy download en_core_web_sm
```

## Contributing

Any contribution is more than welcome

```bash
pip install poetry
git clone https://github.com/charlesdedampierre/BunkaTopics.git
cd BunkaTopics

# Create the environment from the .lock file. 
poetry install # This will install all packages in the .lock file inside a virtual environmnet

# Start the environment
poetry shell
```

## Getting Started

| Name  | Link  |
|---|---|
| Visual Topic Modeling With Bunkatopics  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DtPrI82TYepWLoc4RwuQnOqMJb0eWT_t?usp=sharing)  |

## Quick Start

We start by extracting topics from the Twitter Trump Dataset:

```python
from bunkatopics import Bunka
import random
from datasets import load_dataset

dataset = load_dataset("rguo123/trump_tweets")['train']['content']
full_docs = random.sample(dataset, 10000)

```

You can the load any model from langchain. Some of them might be large, please check the langchain [documentation](https://python.langchain.com/en/latest/reference/modules/embeddings.html)

If you want to start with a small model:

```python
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

bunka = Bunka(model_hf=embedding_model)

bunka.fit(full_docs)
df_topics = bunka.get_topics(n_clusters = 20)
```

If you want a bigger LLM Like [Instructor](https://github.com/HKUNLP/instructor-embedding)

```python
from langchain.embeddings import HuggingFaceInstructEmbeddings

embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",
                                                embed_instruction="Embed the documents for visualisation of Topic Modeling on a map : ")

bunka = Bunka(model_hf=embedding_model)

bunka.fit(full_docs)
df_topics = bunka.get_topics(n_clusters = 8)

```

Then, we can visualize

```python
topic_fig = bunka.visualize_topics( width=800, height=800)
topic_fig
```

<img src="images/newsmap.png" width="70%" height="70%" align="center" />

You can get the topics summarized by OpenAI

```python

# then compute the new topics and insert the key
import os
from dotenv import load_dotenv
load_dotenv() # you can add the key to your environment and load it that way

df_clean_names = bunka.get_clean_topic_name(openai_key = os.getenv("OPEN_AI_KEY"))

```

<img src="images/newsmap_clean.png" width="70%" height="70%" align="center" />

The map display the different texts on a 2-Dimensional unsupervised scale. Every region of the map is a topic described by its most specific terms.

```python

bourdieu_fig = bunka.visualize_bourdieu(
        x_left_words=["war"],
        x_right_words=["peace"],
        y_top_words=["men"],
        y_bottom_words=["women"],
        openai_key=os.getenv("OPEN_AI_KEY"),#set to None otherwise
        height=1500,
        width=1500,
        label_size_ratio_label=50,
        display_percent=True,
        clustering=True,
        topic_n_clusters=10,
        topic_terms=5,
        topic_top_terms_overall=500,
        topic_gen_name=True,
    )

```  

The power of this visualisation is to constrain the axis by creating continuums and looking how the data distribute over these continuums. The inspiration is coming from the French sociologist Bourdieu, who projected items on [2 Dimensional maps](https://www.politika.io/en/notice/multiple-correspondence-analysis).

<img src="images/bourdieu.png" width="70%" height="70%" align="center" />

```python

dimension_fig = bunka.get_dimensions(dimensions=[
                            "Happiness",
                            "Sadness",
                            "Anger",
                            "Love",
                            "Surprise",
                            "Fear",
                            "Excitement",
                            "Disgust",
                            "Confusion",
                            "Gratitude",
                        ])

```

<img src="images/dimensions.png" width="50%" height="50%" align="center" />

## Multilanguage

The package use Spacy to extract meaningfull terms for the topic represenation.

If you wish to change language to french, first, download the corresponding spacy model:

```bash
python -m spacy download fr_core_news_lg
```

```python
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v2")

bunka = Bunka(model_hf=embedding_model, language = 'fr_core_news_lg')

bunka.fit(full_docs)
df_topics = bunka.get_topics(n_clusters = 20)
```  

## Functionality

Here are all the things you can do with Bunkatopics

### Common

Below, you will find an overview of common functions in BERTopic.

| Method | Code  |
|-----------------------|---|
| Fit the model    |  `.fit(docs)` |
| Fit the model and get the topics  |  `.fit_transform(docs)` |
| Acces the topics   | `.get_topics(n_clusters=10)`  |
| Access the top documents per topic    |  `.get_clean_topic_name()` |
| Access the distribution of topics   |  `.get_topic_repartition()` |
| Visualize the topics on a Map |  `.visualize_topics()` |
| Visualize the topics on Natural Language Supervised axis | `.visualize_bourdieu()` |
| Access the Coherence of Topics |  `.get_topic_coherence()` |
| Get the closest documents to your search | `.search('politics')` |

### Attributes

You can access several attributes

| Attribute | Description |
|------------------------|---------------------------------------------------------------------------------------------|
| `.docs`               | The documents stores as a Document pydantic model |
| `.topics` | The Topics stored as a Topic pydantic model. |
