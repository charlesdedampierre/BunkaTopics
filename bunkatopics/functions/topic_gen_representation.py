import typing as t

import openai
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm

from bunkatopics.datamodel import Document, Topic

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
promp_template_topics_terms = """I have a topic that is described the following keywords: 
{terms}

Here are some examples of documents separated by || in the topic:
{documents}:

Based on the information about the topic above, please create a short label that encompassess the meaning of the topic.

Only give the name of the topic and nothing else:

Topic Name:"""


TERM_ID = str


def get_clean_topic(
    generative_model,
    specific_terms: t.List[str],
    specific_documents: t.List[str],
    top_doc: int = 2,
):
    PROMPT_TOPICS = ChatPromptTemplate.from_template(promp_template_topics_terms)

    specific_documents = specific_documents[:top_doc]
    topic_chain = LLMChain(llm=generative_model, prompt=PROMPT_TOPICS)
    clean_topic_name = topic_chain(
        {
            "terms": ", ".join(specific_terms),
            "documents": " \n".join(specific_documents),
        }
    )

    clean_topic_name = clean_topic_name["text"]

    return clean_topic_name


def get_clean_topic_all(
    generative_model, topics: t.List[Topic], docs: t.List[Document], top_doc: int = 3
) -> t.List[Topic]:
    df = get_df_prompt(topics, docs, top_doc)
    topic_ids = list(df["topic_id"])
    specific_terms = list(df["keywords"])
    top_doc_contents = list(df["content"])

    final_dict = {}
    pbar = tqdm(total=len(topic_ids), desc="Creating new labels for clusters")
    for topic_ic, x, y in zip(topic_ids, specific_terms, top_doc_contents):
        clean_topic_name = get_clean_topic(
            generative_model, specific_terms=x, specific_documents=y
        )
        final_dict[topic_ic] = clean_topic_name
        pbar.update(1)

    for topic in topics:
        topic.name = final_dict.get(topic.topic_id)

    return topics


def get_df_prompt(
    topics: t.List[Topic], docs: t.List[Document], top_doc: int = 3
) -> pd.DataFrame:
    """
    get a dataframe to input the prompt


    """
    df_for_prompt = {
        "topic_id": [x.topic_id for x in topics],
        "doc_id": [x.top_doc_id for x in topics],
    }

    df_for_prompt = pd.DataFrame(df_for_prompt)
    df_for_prompt = df_for_prompt.explode("doc_id")

    df_doc = pd.DataFrame(
        {
            "doc_id": [x.doc_id for x in docs],
            "content": [x.content for x in docs],
        }
    )

    df_for_prompt = pd.merge(df_for_prompt, df_doc, on="doc_id")
    df_for_prompt = df_for_prompt.groupby("topic_id")["content"].apply(
        lambda x: list(x)[:top_doc]
    )

    df_keywords = pd.DataFrame(
        {
            "topic_id": [x.topic_id for x in topics],
            "keywords": [x.name.split(" | ") for x in topics],
        }
    )

    df_for_prompt = pd.merge(df_keywords, df_for_prompt, on="topic_id")

    return df_for_prompt
