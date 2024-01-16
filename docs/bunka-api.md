# bunka-api

## `bourdieu_api`

Compute Bourdieu dimensions and topics for a list of documents.

### Args

- `generative_model` (object): The generative AI model.
- `embedding_model` (object): The embedding model.
- `docs` (list of `Document` objects): List of documents.
- `terms` (list of `Term` objects): List of terms.
- `bourdieu_query` (BourdieuQuery object, optional): BourdieuQuery object.
- `topic_param` (TopicParam object, optional): TopicParam object.
- `generative_ai_name` (bool, optional): Whether to generate AI-generated topic names.
- `topic_gen_param` (TopicGenParam object, optional): TopicGenParam object.
- `min_count_terms` (int, optional): Minimum term count.

### Returns

- `Tuple` of lists containing processed documents and topics.

### Example

```python
from bunkatopics import Bunka
from langchain.embeddings import HuggingFaceEmbeddings
import random
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

# Extract Data
dataset = load_dataset("rguo123/trump_tweets")["train"]["content"]
docs = random.sample(dataset, 1000)

# Chose an embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Bunka
bunka = Bunka(language='english')  # Bunka is multilingual. You need to choose a multilingual embedding model as well

# Fit Bunka to your documents
bunka.fit(docs)

# Perform initial topic modeling
bunka.get_topics(n_clusters=10)

# Choose an LLM to summarize the topics so that they are clearer
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Using Mistral Model from Huggingface hub
llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token="HF_TOKEN",
)

bunka.get_clean_topic_name(llm=llm)

# Output the topics
print(df_topics_clean)

# Visualize the topics
bunka.visualize_topics(width=800, height=800)
