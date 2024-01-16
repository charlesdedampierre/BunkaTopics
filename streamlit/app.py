import os
import random

import pandas as pd
from dotenv import load_dotenv

import streamlit as st
from bunkatopics import Bunka

from langchain.llms import HuggingFaceHub
import sys

sys.path.append("../")


load_dotenv()

repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
)


# Define a Streamlit app
st.title("Topic Modeling with Bunka")

# Upload CSV file
csv_file = st.file_uploader(
    "Please upload a CSV file. Bunka recommends sampling a subset of 5000 rows for analysis.",
    type=["csv"],
)

gen_ai = True


# Initialize Bunka and fit it with the text data (cache the fit operation)
@st.cache_resource
def fit_bunka(full_docs):
    bunka.fit(full_docs)
    return bunka


if csv_file is not None:
    # Read CSV into a DataFrame

    df = pd.read_csv(csv_file)

    # Instruct the user to choose a column
    st.write("Please choose a column from the following dropdown:")

    # Create a dropdown menu to select the column
    selected_column = st.selectbox("Select a tesxt column:", df.columns.tolist())

    # Extract text from the selected column
    text_data = df[selected_column].tolist()

    # Specify the desired sample size
    sample_size = 3000

    # Check if the sample size is greater than 500 before sampling
    if sample_size > 500:
        full_docs = random.sample(text_data, sample_size)
    else:
        # Handle the case where sample_size is not greater than 500
        # For example, set full_docs to the entire text_data or another appropriate action.
        full_docs = text_data

    bunka = Bunka()

    bunka = fit_bunka(full_docs)

    st.subheader("Topic Modeling Visualization")
    num_clusters = 5
    df_topics = bunka.get_topics(n_clusters=num_clusters)
    if gen_ai:
        df_clean_names = bunka.get_clean_topic_name(llm=llm)

    # Visualize topics
    topic_fig = bunka.visualize_topics(width=800, height=800)
    st.plotly_chart(topic_fig)

    # Add a section for customizing the visualize_bourdieu parameters
    st.sidebar.title("Customize visualize_bourdieu Parameters")

    x_left_words = st.sidebar.text_input("x_left_words (comma-separated)", "war")
    x_right_words = st.sidebar.text_input("x_right_words (comma-separated)", "peace")
    y_top_words = st.sidebar.text_input("y_top_words (comma-separated)", "men")
    y_bottom_words = st.sidebar.text_input("y_bottom_words (comma-separated)", "women")

    # Display the visualize_bourdieu results
    if st.sidebar.button("Visualize Bourdieu"):
        bunka = fit_bunka(full_docs)
        bourdieu_fig = bunka.visualize_bourdieu(
            llm,
            x_left_words=x_left_words.split(","),
            x_right_words=x_right_words.split(","),
            y_top_words=y_top_words.split(","),
            y_bottom_words=y_bottom_words.split(","),
            height=800,
            width=800,
            display_percent=True,
            clustering=True,
            topic_n_clusters=10,
            topic_terms=5,
            topic_top_terms_overall=500,
            topic_gen_name=False,
        )
        st.subheader("Bourdieu Visualization")
        st.plotly_chart(bourdieu_fig)
