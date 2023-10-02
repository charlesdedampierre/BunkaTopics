import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bunkatopics.datamodel import Document
from bunkatopics.functions.utils import specificity


def plot_specific_terms(
    docs: t.List[Document],
    left_words=["hate", "pain"],
    right_words=["love", "good"],
    id="emotion",
    ngrams=[2],
    quantile=0.80,
    top_n=20,
):
    distances = [
        x.distance
        for doc in docs
        for x in doc.bourdieu_dimensions
        if x.continuum.id == id
    ]
    doc_id = [x.doc_id for x in docs]
    content = [x.content for x in docs]

    df_distances = pd.DataFrame(
        {"doc_id": doc_id, "distances": distances, "content": content}
    )

    df_terms = pd.DataFrame(
        {
            "doc_id": [x.doc_id for x in docs],
            "term_id": [x.term_id for x in docs],
        }
    )

    df_terms = df_terms.explode("term_id").reset_index(drop=True)
    df_terms_docs = pd.merge(df_distances, df_terms, on="doc_id")
    df_terms_docs["group"] = np.nan

    df_terms_docs["group"][
        df_terms_docs["distances"] > df_distances["distances"].quantile(quantile)
    ] = "1"
    df_terms_docs["group"][
        df_terms_docs["distances"] <= df_distances["distances"].quantile(1 - quantile)
    ] = "0"
    df_terms_docs = df_terms_docs.sort_values("distances", ascending=False)
    df_terms_docs = (
        df_terms_docs.groupby(["group", "term_id"])["doc_id"].count().reset_index()
    )

    df_terms_docs["lenght"] = df_terms_docs["term_id"].apply(
        lambda x: len(x.split(" "))
    )
    df_terms_docs = df_terms_docs[df_terms_docs["lenght"].isin(ngrams)]

    df_terms_docs["term_lenght"] = df_terms_docs["term_id"].apply(lambda x: len(x))
    df_terms_docs = df_terms_docs[df_terms_docs["term_lenght"] > 1]

    edge = specificity(df_terms_docs, X="group", Y="term_id", Z="doc_id", top_n=100)

    # Identify terms that appear in both group 0 and group 1
    common_terms = set(edge[edge["group"] == "0"]["term_id"]).intersection(
        set(edge[edge["group"] == "1"]["term_id"])
    )

    # Filter the DataFrame to keep only terms that are not in common_terms
    edge = edge[~edge["term_id"].isin(common_terms)]
    edge["specificity_score"] = pd.qcut(
        edge["specificity_score"].rank(method="first"), q=20, labels=np.arange(1, 21)
    )
    edge["specificity_score"] = edge["specificity_score"].astype(int)
    edge = edge.groupby("group").head(top_n).reset_index(drop=True)

    # Separate data for group=0 and group=1
    group_0 = edge[edge["group"] == "0"]
    group_0["specificity_score"] = -group_0["specificity_score"]
    group_0 = group_0.sort_values("specificity_score", ascending=False)

    group_1 = edge[edge["group"] == "1"]
    group_1 = group_1.sort_values("specificity_score")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        group_0["term_id"],
        group_0["specificity_score"],
        color="blue",
        label="Group 0",
        alpha=0.7,
    )

    # Plot horizontal bars for group=1 on the right
    ax.barh(
        group_1["term_id"],
        group_1["specificity_score"],
        color="red",
        label="Group 1",
        alpha=0.7,
    )

    # Set labels and title
    ax.set_xlabel("Specificity Score")
    ax.set_ylabel("")

    left = " ".join(left_words)
    right = " ".join(right_words)

    title = "<" + right + " | " + left + ">"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
