import pandas as pd
from .basic_class import BasicSemantics
from .specificity import specificity
import umap
from sklearn.cluster import KMeans
from .utils import wrap_by_word
import plotly.express as px


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
            reduction=5,
            multiprocessing=multiprocessing,
        )

    def get_clusters(
        self,
        topic_number: int = 20,
        top_terms: int = 10,
        term_type: str = "lemma",
        top_terms_included: int = 100,
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

        self.data["cluster"] = (
            KMeans(n_clusters=topic_number)
            .fit(self.docs_embeddings)
            .labels_.astype(str)
        )

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
            self.data[["cluster"]],
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
            self.data[["cluster"]]
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
            self.data[["cluster"]].reset_index(), topics, on="cluster"
        )

        self.df_topics_names["cluster_name_number"] = (
            self.df_topics_names["cluster"]
            + " - "
            + self.df_topics_names["cluster_name"]
        )

        self.df_topics_names = self.df_topics_names.set_index(self.index_var)

        return self.topics

    def visualize_clusters(self, search: str = None, width=1000, height=1000):
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
            self.data,
            left_index=True,
            right_index=True,
        )

        if search is not None:
            df_search = self.data[self.text_var].reset_index()
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
        self.embeddings_2d = umap.UMAP(
            n_components=2, random_state=42, verbose=True
        ).fit_transform(res[[0, 1, 2, 3, 4]])

        res["dim_1"] = self.embeddings_2d[:, 0]
        res["dim_2"] = self.embeddings_2d[:, 1]

        res[self.text_var] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))
        res["cluster_label"] = (
            res["cluster"].astype(object) + " - " + res["cluster_name"]
        )

        res["cluster_size"] = (
            res["cluster"].astype(str) + " | " + res["topic_size"].astype(str)
        )

        self.df_fig = res.reset_index(drop=True)

        # Compute centroids to add the cluster label on it
        centroids_emb = self.df_fig[["dim_1", "dim_2", "cluster_name_number"]]
        centroids_emb = (
            centroids_emb.groupby("cluster_name_number").mean().reset_index()
        )
        centroids_emb.columns = ["centroid_name", "dim_1", "dim_2"]

        df_fig_centroids = pd.concat([self.df_fig, centroids_emb])
        df_fig_centroids["centroid_name"] = df_fig_centroids["centroid_name"].fillna(
            " "
        )
        df_fig_centroids["cluster_size"] = df_fig_centroids["cluster_size"].fillna(
            "centroids"
        )

        fig = px.scatter(
            df_fig_centroids,
            x="dim_1",
            y="dim_2",
            color=color,
            text="centroid_name",
            hover_data=[self.text_var],
            width=width,
            height=height,
        )

        return fig
