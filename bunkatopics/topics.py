import pandas as pd
from .basic_class import BasicSemantics
from .specificity import specificity
from sklearn.cluster import KMeans
from .density_plot import get_density_plot
import numpy as np
from .folding_utils import index_data
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn import metrics

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class BunkaTopics(BasicSemantics):
    def __init__(
        self,
        data,
        text_var,
        index_var,
        extract_terms=True,
        terms_embeddings=True,
        docs_embeddings=True,
        sample_size_terms=500,
        terms_limit=500,
        terms_ents=True,
        terms_ngrams=(1, 2),
        terms_ncs=True,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        embeddings_model="distiluse-base-multilingual-cased-v1",
        multiprocessing=True,
        language="en",
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
        reduction=5,
    ) -> None:

        BasicSemantics.__init__(
            self,
            data=data,
            text_var=text_var,
            index_var=index_var,
            terms_path=terms_path,
            terms_embeddings_path=terms_embeddings_path,
            docs_embeddings_path=docs_embeddings_path,
        )

        BasicSemantics.fit(
            self,
            extract_terms=extract_terms,
            terms_embeddings=terms_embeddings,
            docs_embeddings=docs_embeddings,
            sample_size_terms=sample_size_terms,
            terms_limit=terms_limit,
            terms_ents=terms_ents,
            terms_ngrams=terms_ngrams,
            terms_ncs=terms_ncs,
            terms_include_pos=terms_include_pos,
            terms_include_types=terms_include_types,
            embeddings_model=embeddings_model,
            language=language,
            reduction=reduction,
            multiprocessing=multiprocessing,
        )

    def get_clusters(
        self,
        topic_number=20,
        top_terms=4,
        term_type="lemma",
        top_terms_included=1000,
        out_terms=["juif"],
        ngrams=(1, 2),
    ):

        data_clusters = self.data.copy()
        df_index_extented = self.df_terms_indexed.reset_index().copy()
        terms = self.terms[self.terms["ngrams"].isin(ngrams)]

        if out_terms is not None:
            terms = terms[~terms[term_type].isin(out_terms)]

        kmeans = KMeans(n_clusters=topic_number, random_state=42)
        data_clusters["cluster"] = kmeans.fit(self.docs_embeddings.values).labels_

        df_index_extented = df_index_extented.explode("text").reset_index(drop=True)

        df_index_extented = pd.merge(
            df_index_extented,
            terms.reset_index().head(top_terms_included),
            on="text",
        )
        df_index_extented = df_index_extented.set_index(self.index_var)

        # Get the Topics Names
        df_clusters = pd.merge(
            data_clusters[["cluster"]],
            df_index_extented,
            left_index=True,
            right_index=True,
        )

        _, _, edge = specificity(
            df_clusters, X="cluster", Y=term_type, Z=None, top_n=top_terms
        )

        topics = (
            edge.groupby("cluster")[term_type]
            .apply(lambda x: " | ".join(x))
            .reset_index()
        )
        topics = topics.rename(columns={term_type: "cluster_name"})

        # Get the Topics Size
        topic_size = (
            data_clusters[["cluster"]]
            .reset_index()
            .groupby("cluster")[self.index_var]
            .count()
            .reset_index()
        )

        topic_size.columns = ["cluster", "topic_size"]
        topics = pd.merge(topics, topic_size, on="cluster")
        topics = topics.sort_values("topic_size", ascending=False).reset_index(
            drop=True
        )
        topics["percent"] = round(
            topics["topic_size"] / topics["topic_size"].sum() * 100, 1
        )

        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=[0, 1])
        centroids["cluster"] = centroids.index

        self.topics = pd.merge(topics, centroids, on="cluster")

        data_clusters = data_clusters[["cluster"]]
        data_clusters = pd.merge(
            data_clusters, self.docs_embeddings, left_index=True, right_index=True
        )
        self.data_clusters = pd.merge(
            data_clusters, topics[["cluster", "cluster_name"]], on="cluster"
        )

        self.data_clusters.index = self.data.index

        return self.topics

    def visualize_clusters(
        self, width: int = 1000, height: int = 1000, sizes=None, colors=None
    ):

        fig = get_density_plot(
            x=list(self.data_clusters[0]),
            y=list(self.data_clusters[1]),
            texts=list(self.data[self.text_var]),
            clusters=list(self.data_clusters["cluster_name"]),
            x_centroids=list(self.topics[0]),
            y_centroids=list(self.topics[1]),
            label_centroids=list(self.topics["cluster_name"]),
            width=width,
            height=height,
            sizes=sizes,
            colors=colors,
        )

        return fig

    def get_specific_documents_per_cluster(
        self, top_n=10, top_type="terms_based", pop_var="Times Cited, WoS Core"
    ):
        """Extract the top documents per clusters based on two rules: (top_type: terms_based)
        - either the documents with the msot specific terms in it: (top_type: pop_based)
        - the most popular documents
        """

        if top_type == "terms_based":

            new_topics = self.topics.copy()
            new_topics["text"] = new_topics["cluster_name"].apply(
                lambda x: x.split(" | ")
            )
            new_topics = new_topics.explode("text")

            df_indexed = self.df_terms_indexed.copy().reset_index()
            df_indexed = df_indexed.explode("text")
            df_indexed = pd.merge(
                df_indexed,
                self.data_clusters[["cluster"]].reset_index(),
                on=self.index_var,
            )

            top_doc = pd.merge(df_indexed, new_topics, on=["text", "cluster"])
            top_doc = (
                top_doc.groupby([self.index_var, "cluster"])["text"]
                .count()
                .reset_index()
            )

            top_doc = top_doc.sort_values(["cluster", "text"], ascending=(False, False))
            top_doc = top_doc.groupby("cluster").head(top_n).reset_index(drop=True)
            top_doc = pd.merge(self.topics, top_doc, on="cluster")
            top_doc = top_doc.rename(columns={"text": "count_terms"})

        elif top_type == "pop_based":

            df_popularity = self.data_clusters[[pop_var] + ["cluster"]]
            df_popularity = df_popularity.sort_values(
                ["cluster"] + [pop_var], ascending=(False, False)
            )
            df_popularity = df_popularity.groupby("cluster").head(top_n).reset_index()
            df_popularity = pd.merge(df_popularity, self.topics, on="cluster")
            top_doc = df_popularity.copy()

        return top_doc

    def get_folding(self, dictionnary):
        """category	term
        0	negative	hate
        1	negative	violence
        2	negative	pain
        3	negative	negative
        4	positive	good
        """

        df_indexed = pd.DataFrame()

        for cat in set(dictionnary.category):

            dictionnary_filtered = dictionnary[dictionnary["category"] == cat][
                "term"
            ].to_list()

            df_indexed_filtered = index_data(
                dictionnary_filtered, self.data[self.text_var].to_list()
            )
            df_indexed_filtered.columns = [self.text_var, "term", "sent_location"]
            df_indexed_filtered = pd.merge(
                df_indexed_filtered,
                self.data[[self.text_var]].reset_index(),
                on=self.text_var,
            )

            df_indexed_filtered["label"] = cat
            df_indexed = df_indexed.append(df_indexed_filtered)

        # Merge with embeddings

        df_embeddings = self.docs_embeddings.reset_index()
        df_embeddings = df_embeddings.rename(columns={"index": self.index_var})
        df_final = pd.merge(df_embeddings, df_indexed, on=self.index_var)

        emb_columns = list(np.arange(self.reduction))

        # Train self
        data_self = df_final[emb_columns + ["label"]]
        data_self.index = df_final[self.index_var]

        df_train, df_test = train_test_split(data_self, test_size=0.3, random_state=42)

        X_train = df_train.drop("label", 1)
        X_test = df_test.drop("label", 1)

        y_train = df_train["label"].to_list()
        y_test = df_test["label"].to_list()

        classifier = OneVsRestClassifier(XGBClassifier())
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        met = metrics.classification_report(y_test, y_pred, digits=3)

        # Predict for the whiole dataset
        df_folding = pd.DataFrame(
            classifier.predict_proba(self.docs_embeddings), columns=classifier.classes_
        )
        df_folding["label"] = classifier.predict(self.docs_embeddings)
        df_folding.index = self.docs_embeddings.index

        df_folding = pd.merge(
            df_folding, self.data[[self.text_var]], left_index=True, right_index=True
        )
        df_folding = pd.merge(
            df_folding, self.docs_embeddings, left_index=True, right_index=True
        )

        return df_folding, met
