import random
import typing as t

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from bunkatopics.datamodel import (BourdieuDimension, BourdieuQuery,
                                   ContinuumDimension, Document, Term, Topic,
                                   TopicGenParam, TopicParam)
from bunkatopics.functions.topic_document import get_top_documents
from bunkatopics.functions.topic_gen_representation import get_clean_topic_all
from bunkatopics.functions.topics_modeling import get_topics
from bunkatopics.visualisation.visu_utils import wrap_by_word

pd.options.mode.chained_assignment = None


def bourdieu_api(
    embedding_model,
    docs: t.List[Document],
    terms: t.List[Term],
    generative_model=None,
    bourdieu_query: BourdieuQuery = BourdieuQuery(),
    topic_param: TopicParam = TopicParam(),
    generative_ai_name=False,
    topic_gen_param: TopicGenParam = TopicGenParam(),
) -> (t.List[Document], t.List[Topic]):
    # Reset
    for doc in docs:
        doc.bourdieu_dimensions = []

    # Compute Continuums
    new_docs = get_continuum(
        embedding_model,
        docs,
        cont_name="cont1",
        left_words=bourdieu_query.x_left_words,
        right_words=bourdieu_query.x_right_words,
    )
    bourdieu_docs = get_continuum(
        embedding_model,
        new_docs,
        cont_name="cont2",
        left_words=bourdieu_query.y_top_words,
        right_words=bourdieu_query.y_bottom_words,
    )

    # There are two coordinates
    df_bourdieu = pd.DataFrame(
        [
            {
                "doc_id": x.doc_id,
                "coordinates": [y.distance for y in x.bourdieu_dimensions],
                "names": [y.continuum.id for y in x.bourdieu_dimensions],
            }
            for x in bourdieu_docs
        ]
    )

    df_bourdieu = df_bourdieu.explode(["coordinates", "names"])

    df_bourdieu_pivot = df_bourdieu[["doc_id", "coordinates", "names"]]
    df_bourdieu_pivot = df_bourdieu_pivot.pivot(
        index="doc_id", columns="names", values="coordinates"
    )

    # Add to the bourdieu_docs

    df_outsides = df_bourdieu_pivot.reset_index()
    df_outsides["cont1"] = df_outsides["cont1"].astype(
        float
    )  # Cont1 is the default name
    df_outsides["cont2"] = df_outsides["cont2"].astype(float)

    x_values = df_outsides["cont1"].values
    y_values = df_outsides["cont2"].values

    distances = np.sqrt(x_values**2 + y_values**2)
    circle_radius = max(df_outsides.cont1) * bourdieu_query.radius_size

    df_outsides["distances"] = distances
    df_outsides["outside"] = "0"
    df_outsides["outside"][df_outsides["distances"] >= circle_radius] = "1"

    outside_ids = list(df_outsides["doc_id"][df_outsides["outside"] == "1"])

    bourdieu_docs = [x for x in bourdieu_docs if x.doc_id in outside_ids]
    bourdieu_dict = df_bourdieu_pivot.to_dict(orient="index")

    for doc in bourdieu_docs:
        doc.x = bourdieu_dict.get(doc.doc_id)["cont1"]
        doc.y = bourdieu_dict.get(doc.doc_id)["cont2"]

    bourdieu_topics = get_topics(
        docs=bourdieu_docs,
        terms=terms,
        n_clusters=topic_param.n_clusters,
        ngrams=topic_param.ngrams,
        name_lenght=topic_param.name_lenght,
        top_terms_overall=topic_param.top_terms_overall,
    )

    bourdieu_docs = get_top_documents(bourdieu_docs, bourdieu_topics, ranking_terms=20)

    if generative_ai_name:
        bourdieu_topics: t.List[Topic] = get_clean_topic_all(
            generative_model,
            bourdieu_topics,
            bourdieu_docs,
            language=topic_gen_param.language,
            context=topic_gen_param.context,
            use_doc=topic_gen_param.use_doc,
        )

    return (bourdieu_docs, bourdieu_topics)


def visualize_bourdieu(
    embedding_model,
    generative_model,
    docs: t.List[Document],
    terms: t.List[Term],
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
    use_doc_gen_topic: bool = False,
    gen_topic_language: str = "english",
    label_size_ratio_label: int = 50,
    topic_top_terms_overall: int = 500,
    manual_axis_name: dict = None,
    radius_size: float = 0.3,
    convex_hull: bool = True,
):
    # Reset
    for doc in docs:
        doc.bourdieu_dimensions = []

    # Compute Continuums
    new_docs = get_continuum(
        embedding_model,
        docs,
        cont_name="cont1",
        left_words=x_left_words,
        right_words=x_right_words,
    )
    new_docs = get_continuum(
        embedding_model,
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

    # Remove the data inside the radius of 1/3 of max because central data does not mean mucj
    df_fig["cont1"] = df_fig["cont1"].astype(float)
    df_fig["cont2"] = df_fig["cont2"].astype(float)

    import numpy as np

    x_values = df_fig["cont1"].values
    y_values = df_fig["cont2"].values

    distances = np.sqrt(x_values**2 + y_values**2)

    circle_radius = max(df_fig.cont1) * radius_size

    df_fig["distances"] = distances
    df_fig["outside"] = "0"
    df_fig["outside"][df_fig["distances"] >= circle_radius] = "1"

    outside_ids = list(df_fig["doc_id"][df_fig["outside"] == "1"])

    df_fig = df_fig[df_fig["doc_id"].isin(outside_ids)]

    df_fig = pd.merge(df_content, df_fig, on="doc_id")
    df_fig["Text"] = df_fig["content"].apply(lambda x: wrap_by_word(x, 10))

    x_axis_name = list(dict_bourdieu.keys())[0]
    y_axis_name = list(dict_bourdieu.keys())[1]

    x_left_words = dict_bourdieu[x_axis_name]["left_words"]
    x_right_words = dict_bourdieu[x_axis_name]["right_words"]
    y_top_words = dict_bourdieu[y_axis_name]["left_words"]
    y_bottom_words = dict_bourdieu[y_axis_name]["right_words"]

    fig = go.Figure(
        go.Histogram2dContour(
            x=df_fig[x_axis_name],
            y=df_fig[y_axis_name],
            colorscale="delta",
            showscale=False,
        ),
    )

    scatter_fig = px.scatter(
        df_fig,
        x=x_axis_name,
        y=y_axis_name,
        color="outside",
        color_discrete_map={"1": "white", "0": "grey"},
        hover_data=["Text"],
        template="simple_white",
        height=height,
        width=width,
        opacity=0.3,
        # title="Bourdieu Plot"
        # color_discrete_sequence=["blue"],
    )

    for trace in scatter_fig.data:
        fig.add_trace(trace)

    # Set the axis to the max value to get a square
    max_val = max(
        abs(min(df_fig[y_axis_name])),
        abs(max(df_fig[y_axis_name])),
        abs(max(df_fig[x_axis_name])),
        abs(min(df_fig[x_axis_name])),
    )

    # Add axis lines for x=0 and y=0
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        # y0=-max_val,
        # y1=max_val,
        y0=min(df_fig[y_axis_name]),
        y1=max(df_fig[y_axis_name]),
        line=dict(color="white", width=3),  # Customize line color and width
    )

    fig.add_shape(
        type="line",
        x0=min(df_fig[x_axis_name]),
        x1=max(df_fig[x_axis_name]),
        # x0=-max_val,
        # x1=max_val,
        y0=0,
        y1=0,
        line=dict(color="white", width=3),  # Customize line color and width
    )

    fig.update_layout(
        font_size=25,
        width=width,
        height=height,
        margin=dict(
            t=width / 50,
            b=width / 50,
            r=width / 50,
            l=width / 50,
        ),
        # title=dict(font=dict(size=width / 40)),
    )

    fig.update_layout(showlegend=False)
    """
    histogram2d_contour = go.Figure(
        go.Histogram2dContour(
            x=df_fig[x_axis_name],
            y=df_fig[y_axis_name],
            colorscale="delta",
            showscale=False,
        ),
    )
    fig.add_trace(histogram2d_contour.data[0])

    scatter_fig = px.scatter(
        df_fig,
        x=x_axis_name,
        y=y_axis_name,
        color="outside",
        color_discrete_map={"1": "white", "0": "grey"},
        hover_data=["Text"],
        template="simple_white",
        height=height,
        width=width,
        opacity=0.3,
        # title="Bourdieu Plot"
        # color_discrete_sequence=["blue"],
    )

    for trace in scatter_fig.data:
        fig.add_trace(trace)

    """

    """


    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=True,
        zerolinecolor="white",
        zerolinewidth=2,
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=True,
        zerolinecolor="white",
        zerolinewidth=2,
    )

    """

    if manual_axis_name is None:
        y_top_name = y_top_words
        y_bottom_name = y_bottom_words
        x_left_name = x_left_words
        x_right_name = x_right_words

    else:
        y_top_name = manual_axis_name["y_top_name"]
        y_bottom_name = manual_axis_name["y_bottom_name"]
        x_left_name = manual_axis_name["x_left_name"]
        x_right_name = manual_axis_name["x_right_name"]

    fig.update_layout(
        annotations=[
            dict(
                x=0,
                # y=max_val,
                y=max(df_fig[y_axis_name]),
                xref="x",
                yref="y",
                text=y_top_name,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=width / label_size_ratio_label, color="white"),
            ),
            dict(
                x=0,
                y=min(df_fig[y_axis_name]),
                # y=-max_val,
                xref="x",
                yref="y",
                text=y_bottom_name,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=width / label_size_ratio_label, color="white"),
            ),
            dict(
                x=max(df_fig[x_axis_name]),
                # x=max_val,
                y=0,
                xref="x",
                yref="y",
                text=x_left_name,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=width / label_size_ratio_label, color="white"),
            ),
            dict(
                x=min(df_fig[x_axis_name]),
                # x=-max_val,
                y=0,
                xref="x",
                yref="y",
                text=x_right_name,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=width / label_size_ratio_label, color="white"),
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

        new_docs = [doc for doc in new_docs if doc.doc_id in outside_ids]

        bourdieu_topics = get_topics(
            docs=new_docs,
            terms=terms,
            n_clusters=topic_n_clusters,
            ngrams=topic_ngrams,
            name_lenght=topic_terms,
            top_terms_overall=topic_top_terms_overall,
        )

        if topic_gen_name:
            # Get top documents for the generative AI query
            new_docs = get_top_documents(new_docs, bourdieu_topics, ranking_terms=20)

            bourdieu_topics = get_clean_topic_all(
                generative_model,
                language=gen_topic_language,
                topics=bourdieu_topics,
                docs=new_docs,
                use_doc=use_doc_gen_topic,
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

        if convex_hull:
            try:
                for topic in bourdieu_topics:
                    # Create a Scatter plot with the convex hull coordinates
                    trace = go.Scatter(
                        x=topic.convex_hull.x_coordinates,
                        y=topic.convex_hull.y_coordinates,  # Assuming y=0 for simplicity
                        mode="lines",
                        name="Convex Hull",
                        line=dict(color="grey"),
                        showlegend=False,
                    )

                    fig.add_trace(trace)
            except:
                pass

    if display_percent:
        # Calculate the percentage for every box

        df_fig_percent = df_fig[df_fig["doc_id"].isin(outside_ids)]

        label_size_ratio_percent = 20
        opacity = 0.4
        case1_count = len(
            df_fig_percent[
                (df_fig_percent["cont1"] < 0) & (df_fig_percent["cont2"] < 0)
            ]
        )
        total_count = len(df_fig_percent)
        case1_percentage = str(round((case1_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=min(df_fig_percent[x_axis_name]),
            y=min(df_fig_percent[y_axis_name]),
            text=case1_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
            xanchor="left",
        )

        case2_count = len(
            df_fig_percent[
                (df_fig_percent["cont1"] < 0) & (df_fig_percent["cont2"] > 0)
            ]
        )
        case2_percentage = str(round((case2_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=min(df_fig_percent[x_axis_name]),
            y=max(df_fig_percent[y_axis_name]),
            text=case2_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
            xanchor="left",
        )

        case3_count = len(
            df_fig_percent[
                (df_fig_percent["cont1"] > 0) & (df_fig_percent["cont2"] < 0)
            ]
        )
        case3_percentage = str(round((case3_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=max(df_fig_percent[x_axis_name]),
            y=min(df_fig_percent[y_axis_name]),
            text=case3_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
            xanchor="left",
        )

        case4_count = len(
            df_fig_percent[
                (df_fig_percent["cont1"] > 0) & (df_fig_percent["cont2"] > 0)
            ]
        )
        case4_percentage = str(round((case4_count / total_count) * 100, 1)) + "%"

        fig.add_annotation(
            x=max(df_fig_percent[x_axis_name]),
            y=max(df_fig_percent[y_axis_name]),
            text=case4_percentage,
            font=dict(
                family="Courier New, monospace",
                size=width / label_size_ratio_percent,
                color="grey",
            ),
            opacity=opacity,
            xanchor="left",
        )

    # Update the x-axis and y-axis labels
    fig.update_xaxes(
        title_text="",
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        showticklabels=False,
        zeroline=True,
        zerolinecolor="white",
        zerolinewidth=2,
    )
    fig.update_yaxes(
        title_text="",
        scaleanchor="x",
        scaleratio=1,
        showgrid=False,
        showticklabels=False,
        zeroline=True,
        zerolinecolor="white",
        zerolinewidth=2,
    )

    return fig, df_bourdieu


def get_continuum(
    embedding_model,
    docs: t.List[Document],
    cont_name: str = "emotion",
    left_words: list = ["hate", "pain"],
    right_words: list = ["love", "good"],
    scale: bool = False,
) -> t.List[Document]:
    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_emb = df_docs[["doc_id", "embedding"]]
    df_emb = df_emb.set_index("doc_id")
    df_emb = pd.DataFrame(list(df_emb["embedding"]))
    df_emb.index = df_docs["doc_id"]

    continuum = ContinuumDimension(
        id=cont_name, left_words=left_words, right_words=right_words
    )

    # Compute the extremity embeddings
    left_embedding = embedding_model.embed_documents(continuum.left_words)
    right_embedding = embedding_model.embed_documents(continuum.right_words)

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
    df_bert = df_bert.iloc[
        -1:,
    ].T
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
    bourdieu_docs = docs.copy()
    for doc in bourdieu_docs:
        res = BourdieuDimension(
            continuum=continuum, distance=distance_dict.get(doc.doc_id)["distance"]
        )
        doc.bourdieu_dimensions.append(res)

    return bourdieu_docs
