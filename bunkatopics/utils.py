import os

import jsonlines
import pandas as pd
import tiktoken

from bunkatopics.datamodel import Document, Term, Topic


def _filter_hdbscan(topics: t.List[Topic], docs: t.List[Document]):
    # Remove for HDBSCAN
    filtered_topics = []
    for topic in topics:
        if topic.topic_id == "bt--1":
            continue
        else:
            filtered_topics.append(topic)

    filtered_docs = []
    for doc in docs:
        if doc.topic_id == "bt--1":
            continue
        else:
            filtered_docs.append(doc)

    return filtered_topics, filtered_docs


def _create_topic_dfs(topics: t.List[Topic], docs: t.List[Document]):
    df_topics = pd.DataFrame.from_records([topic.model_dump() for topic in topics])

    df_topics["percent"] = df_topics["size"] / df_topics["size"].sum()
    df_topics["percent"] = df_topics["percent"] * 100
    df_topics["percent"] = round(df_topics["percent"], 2)
    df_topics = df_topics.rename(columns={"name": "topic_name"})

    df_topics = df_topics[["topic_id", "topic_name", "size", "percent"].copy()]
    df_topics = df_topics.sort_values("size", ascending=False)
    df_topics = df_topics.reset_index(drop=True)

    # extract Dataframe for top documents per topic
    top_docs_topics = [x for x in docs if x.topic_ranking is not None]
    top_docs_topics = pd.DataFrame([x.model_dump() for x in top_docs_topics])
    top_docs_topics["ranking_per_topic"] = top_docs_topics["topic_ranking"].apply(
        lambda x: x.get("rank")
    )
    top_docs_topics = top_docs_topics[
        ["topic_id", "content", "ranking_per_topic", "doc_id"]
    ]
    top_docs_topics = top_docs_topics.sort_values(
        ["topic_id", "ranking_per_topic"], ascending=(True, True)
    )
    top_docs_topics = top_docs_topics.reset_index(drop=True)
    top_docs_topics = pd.merge(
        top_docs_topics, df_topics[["topic_id", "topic_name"]], on="topic_id"
    )

    top_docs_topics = top_docs_topics[
        ["doc_id", "content", "ranking_per_topic", "topic_id", "topic_name"]  # re-order
    ]

    return df_topics, top_docs_topics


class BunkaError(Exception):
    """Custom exception for Bunka-related errors."""

    pass


def save_bunka_models(bunka, path="bunka_dump"):
    os.makedirs(path, exist_ok=True)

    for doc in bunka.docs:
        list_of_floats = [float(value) for value in doc.embedding]
        doc.embedding = list_of_floats

    # Dump the data into JSONL files
    with jsonlines.open(path + "/bunka_docs.jsonl", mode="w") as writer:
        for item in bunka.docs:
            writer.write(item.dict())

    # Dump the data into JSONL files
    with jsonlines.open(path + "/bunka_terms.jsonl", mode="w") as writer:
        for item in bunka.terms:
            writer.write(item.dict())


# Define a function to read documents from a JSONL file
def read_documents_from_jsonl(file_path):
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for item in reader:
            document = Document(**item)
            documents.append(document)
    return documents


def read_terms_from_jsonl(file_path):
    terms = []
    with jsonlines.open(file_path, mode="r") as reader:
        for item in reader:
            term = Term(**item)
            terms.append(term)
    return terms


enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(docs):
    tokens = [enc.encode(x) for x in docs]
    sum_tokens = [len(x) for x in tokens]
    total_number_of_tokens = sum(sum_tokens)
    return total_number_of_tokens
