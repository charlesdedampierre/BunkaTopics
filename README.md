# BunkaTopics

BunkaTopics is a Topic Modeling package that leverages Embeddings and focuses on Topic Representation to extract meaningful and interpretable topics from a list of documents.

## Installation

Intall the package using pip

```bash
pip install bunkatopics
```

## Quick Start with BunkaTopics

```python
from bunkatopics import BunkaTopics
import pandas as pd

data = pd.read_csv('data/imdb.csv', index_col = [0])
data = data.sample(2000, random_state = 42)

# Instantiate the model, extract ther terms and Embed the documents

model = BunkaTopics(data, # dataFrame
                    text_var = 'description', # Text Columns
                    index_var = 'imdb',  # Index Column (Mandatory)
                    extract_terms=True, # extract Terms ?
                    terms_embeddings=True, # extract terms Embeddings?
                    docs_embeddings=True, # extract Docs Embeddings?
                    embeddings_model="distiluse-base-multilingual-cased-v1", # Chose an embeddings Model
                    multiprocessing=True, # Multiprocessing of Embeddings
                    language="en", # Chose between English "en" and French "fr"
                    sample_size_terms = len(data),
                    terms_limit=10000, # Top Terms to Output
                    terms_ents=True, # Extract entities
                    terms_ngrams=(1, 2), # Chose Ngrams to extract
                    terms_ncs=True, # Extract Noun Chunks
                    terms_include_pos=["NOUN", "PROPN", "ADJ"], # Include Part-of-Speech
                    terms_include_types=["PERSON", "ORG"]) # Include Entity Types

# Extract the topics

topics = model.get_clusters(topic_number= 15, # Number of Topics
                    top_terms_included = 1000, # Compute the specific terms from the top n terms
                    top_terms = 5, # Most specific Terms to describe the topics
                    term_type = "lemma", # Use "lemma" of "text"
                    ngrams = [1, 2]) # N-grams for Topic Representation

# Visualize the clusters. It is adviced to choose less that 5 terms - top_terms = 5 - to avoid overchanging the Figure

fig = model.visualize_clusters(search = None, width=1000, height=1000)
fig.show()
```
