import logging
import pandas as pd
from .basic_class import BasicSemantics
from .specificity import specificity
import umap
from sklearn.cluster import KMeans
from .utils import wrap_by_word
import plotly.express as px
from .density_plot import get_density_plot
import hdbscan
from .centroids import find_centroids
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
        topic_number: int = 20,
        top_terms: int = 10,
        term_type: str = "lemma",
        top_terms_included: int = 100,
        clusterer: str = "hdbscan",
        ngrams: list = [1, 2],
    ) -> pd.DataFrame:
        """Get the main topics and the topic representation.

        Parameters
        ----------
        topic_number : int, optional
            Number of topics to get, by default 20
        top_terms : int, optional
            Top terms to describe each topic, by default 10
        term_type : str, optional
            'lemma' or 'text depending if you want the exact text or the lemma', by default "lemma"
        top_terms_included : int, optional
            only select the top n occuring terms, by default 100
        ngrams : list, optional
            Only get the n-grams terms, by default [1, 2]

        Returns
        -------
        _type_
            _description_
        """

        self.data_clusters = self.data.copy()

        if clusterer == "hdbscan":
            self.data_clusters["cluster"] = (
                hdbscan.HDBSCAN().fit(self.docs_embeddings).labels_.astype(str)
            )

            self.data_clusters = self.data_clusters[
                self.data_clusters["cluster"] != "-1"
            ]

        elif clusterer == "kmeans":
            self.kmeans = KMeans(n_clusters=topic_number, random_state=42)
            self.data_clusters["cluster"] = self.kmeans.fit(
                self.docs_embeddings
            ).labels_.astype(str)

        else:
            raise ValueError("Chose between 'kmeans' or 'hdbscan'")

        df_index_extented = self.df_terms_indexed.reset_index().copy()
        df_index_extented = df_index_extented.explode("text").reset_index(drop=True)

        df_index_extented = pd.merge(
            df_index_extented,
            self.terms[self.terms["ngrams"].isin(ngrams)]
            .reset_index()
            .head(top_terms_included),
            on="text",
        )
        df_index_extented = df_index_extented.set_index(self.index_var)

        # Get the Topics Names
        df_clusters = pd.merge(
            self.data_clusters[["cluster"]],
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
            self.data_clusters[["cluster"]]
            .reset_index()
            .groupby("cluster")[self.index_var]
            .count()
            .reset_index()
        )
        topic_size.columns = ["cluster", "topic_size"]

        topics = pd.merge(topics, topic_size, on="cluster")
        topics = topics.sort_values("topic_size", ascending=False)
        self.topics = topics.reset_index(drop=True)

        self.df_topics_names = pd.merge(
            self.data_clusters[["cluster"]].reset_index(), topics, on="cluster"
        )

        self.df_topics_names["cluster_name_number"] = (
            self.df_topics_names["cluster"]
            + " - "
            + self.df_topics_names["cluster_name"]
        )

        self.df_topics_names = self.df_topics_names.set_index(self.index_var)

        return self.topics

    def visualize_clusters(
        self,
        search: str = None,
        scatter_size=None,
        scatter_color=None,
        width=1000,
        height=1000,
        fit_clusters=True,
        density_plot=True,
    ):
        """Visualize the embeddings and the clustering. Search with exact search documents that
        contains your query and visualize it

        Parameters
        ----------
        search : _type_, optional
            _description_, by default None
        """
        res = pd.merge(
            self.docs_embeddings,
            self.df_topics_names,
            left_index=True,
            right_index=True,
        )

        res = pd.merge(
            res.drop("cluster", axis=1),
            self.data_clusters,
            left_index=True,
            right_index=True,
        )

        if search is not None:
            df_search = self.data_clusters[self.text_var].reset_index()
            df_search = df_search[
                df_search[self.text_var].str.contains(search, case=False)
            ]
            df_search = df_search.set_index(self.index_var)
            search_index = list(df_search.index)
            res["search"] = 0
            res["search"][res.index.isin(search_index)] = 1
            # res['search'] = res['search'].astype(object)
            color = "search"

        else:
            color = "cluster_size"

        # if not hasattr(model, "embeddings_2d"):

        len_dim = np.arange(self.reduction)

        """
        if fit_clusters:
            folding = umap.UMAP(n_components=2, random_state=42, verbose=True)
            folding.fit(res[len_dim], res["cluster"].astype(int).to_list())

            self.embeddings_2d = folding.transform(res[len_dim])
        else:
            self.embeddings_2d = umap.UMAP(
                n_components=2, random_state=42, verbose=True
            ).fit_transform(res[len_dim])

        res["dim_1"] = self.embeddings_2d[:, 0]
        res["dim_2"] = self.embeddings_2d[:, 1]

        """

        res[["dim_1"]] = res[0]
        res[["dim_2"]] = res[1]

        res[self.text_var] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))
        res["cluster_label"] = (
            res["cluster"].astype(object) + " - " + res["cluster_name"]
        )

        res["cluster_size"] = (
            res["cluster"].astype(str) + " | " + res["topic_size"].astype(str)
        )

        # self.df_fig = res.reset_index(drop=True)
        self.df_fig = res.copy()

        centroids_emb = pd.DataFrame(
            self.kmeans.cluster_centers_, columns=["dim_1", "dim_2"]
        )
        centroids_emb["cluster"] = centroids_emb.index

        df_centroid = self.df_topics_names.copy()

        df_centroid["cluster"] = df_centroid["cluster_name_number"].apply(
            lambda x: x.split(" - ")[0]
        )
        df_centroid = (
            df_centroid[["cluster_name_number", "cluster"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        centroids_emb = pd.merge(df_centroid, centroids_emb, on="cluster")
        centroids_emb = centroids_emb.drop("cluster", 1)

        centroids_emb = centroids_emb.rename(
            columns={"cluster_name_number": "centroid_name"}
        )

        """
        # Compute centroids to add the cluster label on it
        centroids_emb = self.df_fig[["dim_1", "dim_2", "cluster_name_number"]]
        centroids_emb = (
            centroids_emb.groupby("cluster_name_number").mean().reset_index()
        )
        centroids_emb.columns = ["centroid_name", "dim_1", "dim_2"]
        
        """

        # Erase number at the begining of the cluster name
        centroids_emb["centroid_name"] = centroids_emb["centroid_name"].apply(
            lambda x: x.split(" - ")[1]
        )

        if scatter_size is not None:
            centroids_emb[scatter_size] = 0
            # self.df_fig[scatter_size] = np.log(1 + self.df_fig[scatter_size].fillna(0))
        else:
            pass
            # self.df_fig[scatter_size] = width / 400

        df_fig_centroids = pd.concat([self.df_fig, centroids_emb])
        df_fig_centroids["centroid_name"] = df_fig_centroids["centroid_name"].fillna(
            " "
        )
        df_fig_centroids["cluster_size"] = df_fig_centroids["cluster_size"].fillna(
            "centroids"
        )

        if scatter_color is None:
            colors = None
        else:
            colors = self.df_fig[scatter_color]

        if scatter_size is None:
            sizes = None
        else:
            sizes = self.df_fig[scatter_size]

        if density_plot:
            fig = get_density_plot(
                x=self.df_fig["dim_1"],
                y=self.df_fig["dim_2"],
                texts=self.df_fig[self.text_var],
                clusters=self.df_fig["cluster_label"],
                sizes=sizes,
                colors=colors,
                x_centroids=centroids_emb["dim_1"],
                y_centroids=centroids_emb["dim_2"],
                label_centroids=centroids_emb["centroid_name"],
                width=width,
                height=height,
            )

        else:
            if scatter_size is not None:
                size = df_fig_centroids[scatter_size]
            else:
                size = None

            fig = px.scatter(
                df_fig_centroids,
                x="dim_1",
                y="dim_2",
                color=color,
                size=size,
                text="centroid_name",
                hover_data=[self.text_var],
                width=width,
                height=height,
            )

        return fig

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

    def get_specific_documents_per_cluster(
        self, top_n=10, top_type="terms_based", pop_var="Times Cited, WoS Core"
    ):
        """Extract the top documents per clusters based on two rules: (top_type: terms_based)
        - either the documents with the msot specific terms in it: (top_type: pop_based)
        - the most popular documents
        """

        if top_type == "terms_based":

            new_topics = self.topics.copy()
            new_topics["percent"] = round(
                new_topics["topic_size"] / new_topics["topic_size"].sum() * 100, 0
            )
            new_topics["text"] = new_topics["cluster_name"].apply(
                lambda x: x.split(" | ")
            )
            new_topics = new_topics.explode("text")

            df_indexed = self.df_terms_indexed.copy().reset_index()
            df_indexed = df_indexed.explode("text")
            df_indexed = pd.merge(
                df_indexed, self.data_clusters[["cluster"]].reset_index()
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

        elif top_type == "pop_based":

            df_popularity = self.data_clusters[[pop_var] + ["cluster"]]
            df_popularity = df_popularity.sort_values(
                ["cluster"] + [pop_var], ascending=(False, False)
            )
            df_popularity = df_popularity.groupby("cluster").head(top_n).reset_index()
            df_popularity = pd.merge(df_popularity, self.topics, on="cluster")
            top_doc = df_popularity.copy()

        return top_doc

    def get_centroid_documents(self, top_elements: int = 2) -> pd.DataFrame:
        """Get the centroid documents of the clusters

        Returns
        -------
        pd.DataFrame
            the centroid_docs are separeated by ' || '

        """

        df_centroid = pd.merge(
            self.docs_embeddings,
            self.df_topics_names,
            left_index=True,
            right_index=True,
        )
        df_centroid = pd.merge(
            df_centroid.drop("cluster", axis=1),
            self.data,
            left_index=True,
            right_index=True,
        )

        """df_centroid = df_centroid.rename(
            columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
        )"""

        df_centroid = df_centroid.rename(
            columns={x: f"{x}" for x in range(self.reduction)}
        )

        res = find_centroids(
            df_centroid.reset_index(),
            text_var=self.text_var,
            cluster_var="cluster_name_number",
            top_elements=top_elements,
            dim_lenght=self.reduction,
        )

        return res
