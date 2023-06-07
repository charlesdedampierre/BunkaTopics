[![PyPI - Python](https://img.shields.io/badge/python-v3.9+-blue.svg)](https://pypi.org/project/bertopic/)
[![PyPI - PyPi](https://img.shields.io/pypi/v/bunkatopics)](https://pypi.org/project/bunkatopics/)

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

We start by extracting topics from the well-known 20 newsgroups dataset containing English documents:

```python
from bunkatopics import bunkatopics
from sklearn.datasets import fetch_20newsgroups
import random
 
full_docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
full_docs_random = random.sample(full_docs, 1000)

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
                                                query_instruction="Embed the documents for visualisation of Topic Modeling on a map : ")

bunka = Bunka(model_hf=embedding_model)

bunka.fit(full_docs)
df_topics = bunka.get_topics(n_clusters = 20)
```

Then, we can visualize

```python
topic_fig = bunka.visualize_topics( width=800, height=800)
topic_fig
...
```

The map display the different texts on a 2-Dimensional unsupervised scale. Every region of the map is a topic described by its most specific terms.

<img src="images/newsmap.png" width="70%" height="70%" align="center" />

```python

bourdieu_fig = bunka.visualize_bourdieu(x_left_words=["past"],
                                        x_right_words=["future", "futuristic"],
                                        y_top_words=["politics", "Government"],
                                        y_bottom_words=["cultural phenomenons"],
                                        height=2000,
                                        width=2000)
```  

The power of this visualisation is to constrain the axis by creating continuums and looking how the data distribute over these continuums. The inspiration is coming from the French sociologist Bourdieu, who projected items on [2 Dimensional maps](https://www.politika.io/en/notice/multiple-correspondence-analysis).

<img src="images/bourdieu.png" width="70%" height="70%" align="center" />

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
| Access the top documents per topic    |  `.get_top_documents()` |
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
