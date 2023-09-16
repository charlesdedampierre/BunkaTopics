from bunkatopics.datamodel import Document, Term, ContinuumDimension, BourdieuDimension
from bunkatopics.visualisation.visu_utils import wrap_by_word
import typing as t
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceInstructEmbeddings
from bunkatopics.functions.topic_gen_representation import (
    get_df_prompt,
    get_clean_topics,
)
from bunkatopics.functions.topic_document import get_top_documents
from bunkatopics.functions.topics_modeling import get_topics
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None


def get_continuum(
    model_hf: HuggingFaceInstructEmbeddings,
    docs: t.List[Document],
    cont_name="emotion",
    left_words=["hate", "pain"],
    right_words=["love", "good"],
    scale=False,
):
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

    if scale:
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
    terms: t.List[Term],
    openai_key: str = None,
    x_left_words: t.List[str] = ["war"],
    x_right_words: t.List[str] = ["peace"],
    y_top_words: t.List[str] = ["men"],
    y_bottom_words: t.List[str] = ["women"],
    height: int = 1500,
    width: int = 1500,
    clustering: bool = True,
    topic_gen_name: bool = False,
    topic_n_clusters: int = 5,
    topic_terms: int = 2,
    topic_ngrams: list = [1, 2],
    display_percent: bool = True,
    label_size_ratio_label: int = 50,
    topic_top_terms_overall: int = 500,
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

    df_bourdieu = pd.DataFrame(df_bourdieu)
    df_bourdieu = df_bourdieu.explode(["coordinates", "names"])

    # Filter with only the top and bottom data to avoid getting results too far form the continnuums

    df_content = [{"doc_id": x.doc_id, "content": x.content} for x in new_docs]
    df_content = pd.DataFrame(df_content)

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
                y=max(df_fig[y_axis_name]),
                xref="x",
                yref="y",
                text=y_top_words,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=width / label_size_ratio_label),
            ),
            dict(
                x=0,
                y=min(df_fig[y_axis_name]),
                xref="x",
                yref="y",
                text=y_bottom_words,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=width / label_size_ratio_label),
            ),
            dict(
                x=max(df_fig[x_axis_name]),
                y=0,
                xref="x",
                yref="y",
                text=x_left_words,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=width / label_size_ratio_label),
            ),
            dict(
                x=min(df_fig[x_axis_name]),
                y=0,
                xref="x",
                yref="y",
                text=x_right_words,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=width / label_size_ratio_label),
            ),
        ]
    )

    if clustering:
        df_bourdieu_pivot = df_bourdieu.pivot(
            index="doc_id", columns="names", values="coordinates"
        )
        df_bourdieu_pivot = df_bourdieu_pivot.reset_index()
        df_bourdieu_pivot.columns = ["doc_id", "x", "y"]
        df_bourdieu_pivot = df_bourdieu_pivot.set_index("doc_id")

        dict_doc = df_bourdieu_pivot[["x", "y"]].to_dict("index")

        for doc in new_docs:
            doc.x = dict_doc.get(doc.doc_id)["x"]
            doc.y = dict_doc.get(doc.doc_id)["y"]

        bourdieu_topics = get_topics(
            docs=new_docs,
            terms=terms,
            n_clusters=topic_n_clusters,
            ngrams=topic_ngrams,
            name_lenght=topic_terms,
            top_terms_overall=topic_top_terms_overall,
        )

        if topic_gen_name:
            bourdieu_topics = get_top_documents(
                new_docs, bourdieu_topics, ranking_terms=20, top_docs=5
            )
            df_prompt = get_df_prompt(topics=bourdieu_topics, docs=new_docs)
            bourdieu_topics = get_clean_topics(
                df_prompt, topics=bourdieu_topics, openai_key=openai_key
            )

        label_size_ratio_clusters = 100
        topics_x = [x.x_centroid for x in bourdieu_topics]
        topics_y = [x.y_centroid for x in bourdieu_topics]
        topic_names = [x.name for x in bourdieu_topics]
        topics_name_plotly = [wrap_by_word(x, 7) for x in topic_names]

        # Display Topics
        for x, y, label in zip(topics_x, topics_y, topics_name_plotly):
            fig.add_annotation(
                x=x,
                y=y,
                text=label,
                font=dict(
                    family="Courier New, monospace",
                    size=width / label_size_ratio_clusters,
                    color="red",
                ),
                bordercolor="#c7c7c7",
                borderwidth=width / 1000,
                borderpad=width / 500,
                bgcolor="white",
                opacity=1,
            )

    if display_percent:
        # Calculate the percentage for every box

        label_size_ratio_percent = 20
        opacity = 0.4
        case1_count = len(df_fig[(df_fig["cont1"] < 0) & (df_fig["cont2"] < 0)])
        total_count = len(df_fig)
        case1_percentage = str(round((case1_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=min(df_fig[x_axis_name]),
            y=min(df_fig[y_axis_name]),
            text=case1_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
        )

        case2_count = len(df_fig[(df_fig["cont1"] < 0) & (df_fig["cont2"] > 0)])
        case2_percentage = str(round((case2_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=min(df_fig[x_axis_name]),
            y=max(df_fig[y_axis_name]),
            text=case2_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
        )

        case3_count = len(df_fig[(df_fig["cont1"] > 0) & (df_fig["cont2"] < 0)])
        case3_percentage = str(round((case3_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=max(df_fig[x_axis_name]),
            y=min(df_fig[y_axis_name]),
            text=case3_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
        )

        case4_count = len(df_fig[(df_fig["cont1"] > 0) & (df_fig["cont2"] > 0)])
        case4_percentage = str(round((case4_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=max(df_fig[x_axis_name]),
            y=max(df_fig[y_axis_name]),
            text=case4_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
        )

    return fig, df_bourdieu
