from bunkatopics.datamodel import Document, ContinuumDimension, BourdieuDimension
from bunkatopics.visualisation.visu_utils import wrap_by_word
import typing as t
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from langchain.embeddings import HuggingFaceInstructEmbeddings
from bunkatopics.functions.topics import topic_modeling
from bunkatopics.functions.topic_representation import remove_overlapping_terms


def get_continuum(
    model_hf: HuggingFaceInstructEmbeddings,
    docs: t.List[Document],
    cont_name="emotion",
    left_words=["hate", "pain"],
    right_words=["love", "good"],
) -> pd.DataFrame:
    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_emb = df_docs[["doc_id", "embedding"]]
    df_emb = df_emb.set_index("doc_id")
    df_emb = pd.DataFrame(list(df_emb["embedding"]))
    df_emb.index = df_docs["doc_id"]

    continuum = ContinuumDimension(
        id=cont_name, left_words=left_words, right_words=right_words
    )

    # Compute the extremity embeddings
    left_embedding = model_hf.embed_documents(continuum.left_words)
    right_embedding = model_hf.embed_documents(continuum.right_words)

    left_embedding = pd.DataFrame(left_embedding).mean().values.reshape(1, -1)
    right_embedding = pd.DataFrame(right_embedding).mean().values.reshape(1, -1)

    # Make the difference to get the continnum
    continuum_embedding = left_embedding - right_embedding
    df_continuum = pd.DataFrame(continuum_embedding)
    df_continuum.index = ["distance"]

    # Compute the Cosine Similarity
    full_emb = pd.concat([df_emb, df_continuum])
    df_bert = pd.DataFrame(cosine_similarity(full_emb))

    df_bert.index = full_emb.index
    df_bert.columns = full_emb.index
    df_bert = df_bert.iloc[-1:,].T
    df_bert = df_bert.sort_values("distance", ascending=False).reset_index()
    df_bert = df_bert[1:]
    df_bert = df_bert.rename(columns={"index": "doc_id"})
    final_df = pd.merge(df_bert, df_docs[["doc_id", "content"]], on="doc_id")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    final_df[["distance"]] = scaler.fit_transform(final_df[["distance"]])

    final_df = final_df.set_index("doc_id")
    final_df = final_df[["distance"]]

    distance_dict = final_df.to_dict("index")

    for doc in docs:
        res = BourdieuDimension(
            continuum=continuum, distance=distance_dict.get(doc.doc_id)["distance"]
        )
        doc.bourdieu_dimensions.append(res)

    return docs


def visualize_bourdieu(
    model_hf,
    docs: t.List[Document],
    x_left_words: t.List[str] = ["war"],
    x_right_words: t.List[str] = ["peace"],
    y_top_words: t.List[str] = ["men"],
    y_bottom_words: t.List[str] = ["women"],
    height=1500,
    width=1500,
    clustering=True,
    n_clusters=5,
):
    # Reset
    for doc in docs:
        doc.bourdieu_dimensions = []

    # Compute Continuums
    new_docs = get_continuum(
        model_hf,
        docs,
        cont_name="cont1",
        left_words=x_left_words,
        right_words=x_right_words,
    )
    new_docs = get_continuum(
        model_hf,
        docs,
        cont_name="cont2",
        left_words=y_top_words,
        right_words=y_bottom_words,
    )

    df_names = [
        {
            "names": [y.continuum.id for y in x.bourdieu_dimensions],
            "left_words": [y.continuum.left_words for y in x.bourdieu_dimensions],
            "right_words": [y.continuum.right_words for y in x.bourdieu_dimensions],
        }
        for x in new_docs
    ]
    df_names = pd.DataFrame(df_names)
    df_names = df_names.explode(["names", "left_words", "right_words"])
    df_names["left_words"] = df_names["left_words"].apply(lambda x: "-".join(x))
    df_names["right_words"] = df_names["right_words"].apply(lambda x: "-".join(x))
    df_names = df_names.drop_duplicates()
    df_names = df_names.set_index("names")

    dict_bourdieu = df_names.to_dict(orient="index")

    df_bourdieu = [
        {
            "doc_id": x.doc_id,
            "coordinates": [y.distance for y in x.bourdieu_dimensions],
            "names": [y.continuum.id for y in x.bourdieu_dimensions],
        }
        for x in new_docs
    ]

    df_content = [{"doc_id": x.doc_id, "content": x.content} for x in new_docs]
    df_content = pd.DataFrame(df_content)

    df_bourdieu = pd.DataFrame(df_bourdieu)
    df_bourdieu = df_bourdieu.explode(["coordinates", "names"])

    df_fig = df_bourdieu[["doc_id", "coordinates", "names"]]
    df_fig = df_fig.pivot(index="doc_id", columns="names", values="coordinates")
    df_fig = df_fig.reset_index()
    df_fig = pd.merge(df_content, df_fig, on="doc_id")
    df_fig["content_plotly"] = df_fig["content"].apply(lambda x: wrap_by_word(x, 10))

    x_axis_name = list(dict_bourdieu.keys())[0]
    y_axis_name = list(dict_bourdieu.keys())[1]

    x_left_words = dict_bourdieu[x_axis_name]["left_words"]
    x_right_words = dict_bourdieu[x_axis_name]["right_words"]
    y_top_words = dict_bourdieu[y_axis_name]["left_words"]
    y_bottom_words = dict_bourdieu[y_axis_name]["right_words"]

    fig = px.density_contour(
        df_fig,
        x=x_axis_name,
        y=y_axis_name,
        # hover_data=["content_plotly"],
        template="simple_white",
        height=height,
        width=width,
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=["grey"],
    )

    fig2 = px.scatter(
        df_fig,
        x=x_axis_name,
        y=y_axis_name,
        hover_data=["content_plotly"],
        template="simple_white",
        height=height,
        width=width,
        opacity=0.2,
        title="Bourdieu Plot",
        color_discrete_sequence=["blue"],
    )
    fig.add_traces(fig2.data)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=True)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=True)

    fig.update_layout(
        annotations=[
            dict(
                x=0,
                y=1,
                xref="x",
                yref="y",
                text=y_top_words,
                showarrow=False,
                xanchor="right",
                yanchor="top",
            ),
            dict(
                x=0,
                y=-1,
                xref="x",
                yref="y",
                text=y_bottom_words,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            ),
            dict(
                x=1,
                y=0,
                xref="x",
                yref="y",
                text=x_left_words,
                showarrow=False,
                xanchor="right",
                yanchor="top",
            ),
            dict(
                x=-1,
                y=0,
                xref="x",
                yref="y",
                text=x_right_words,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            ),
        ]
    )

    if clustering:
        # Get the Topic Modeling
        doc_ids = list(df_fig["doc_id"])
        x = df_fig[x_axis_name]
        y = df_fig[y_axis_name]
        docs_terms = [doc.term_id for doc in docs]

        dict_doc_embeddings = {
            doc_id: {"x": x_val, "y": y_val}
            for doc_id, x_val, y_val in zip(doc_ids, x, y)
        }
        dict_doc_terms = {
            doc_id: {"term_id": term} for doc_id, term in zip(doc_ids, docs_terms)
        }
        dict_topics, dict_docs = topic_modeling(
            dict_doc_embeddings,
            dict_doc_terms,
            n_clusters=n_clusters,
        )

        df_topics = pd.DataFrame(dict_topics).T
        df_topics["name"] = df_topics["term_id"].apply(lambda x: x[:5])
        df_topics["name"] = df_topics["name"].apply(
            lambda x: remove_overlapping_terms(x)
        )
        df_topics["name"] = df_topics["name"].apply(lambda x: " | ".join(x))
        df_topics["name_plotly"] = df_topics["name"].apply(lambda x: wrap_by_word(x, 7))

        topics_x = list(df_topics["x_centroid"])
        topics_y = list(df_topics["y_centroid"])
        topics_name_plotly = list(df_topics["name_plotly"])
        label_size_ratio = 100

        for x, y, label in zip(topics_x, topics_y, topics_name_plotly):
            fig.add_annotation(
                x=x,
                y=y,
                text=label,
                font=dict(
                    family="Courier New, monospace",
                    size=width / label_size_ratio,
                    color="red",
                ),
                bordercolor="#c7c7c7",
                borderwidth=width / 1000,
                borderpad=width / 500,
                bgcolor="white",
                opacity=1,
            )

    return fig
