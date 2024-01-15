import sys

sys.path.append("../")

import os
import random

import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

import streamlit as st
from bunkatopics import Bunka

load_dotenv()
# Define a Streamlit app
st.title("Topic Modeling with Bunka")

# Upload CSV file
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

gen_ai = False

# from langchain.llms import OpenAI
# openai_api_key = st.text_input("Enter Your OpenAI API-key")
# generative_model = OpenAI(openai_api_key=openai_api_key)


if csv_file is not None:
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract text from the CSV column named 'text' (adjust column name as needed)
    text_data = df["text"].tolist()

    # Sample a subset of the text data (you can adjust the sample size)
    sample_size = 500
    full_docs = random.sample(text_data, sample_size)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bunka = Bunka(embedding_model=embedding_model)

    bunka.fit(full_docs)

    st.subheader("Topic Modeling Visualization")
    num_clusters = 5
    df_topics = bunka.get_topics(n_clusters=num_clusters)

    # Visualize topics
    topic_fig = bunka.visualize_topics(width=800, height=800)
    st.plotly_chart(topic_fig)
