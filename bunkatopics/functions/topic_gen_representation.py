import typing as t

import openai
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm

from bunkatopics.datamodel import Document, Topic
from langchain.llms import OpenAI
from .prompts import promp_template_topics_terms, promp_template_topics_terms_no_docs

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags


TERM_ID = str


def get_clean_topic(
    generative_model,
    specific_terms: t.List[str],
    specific_documents: t.List[str],
    language="english",
    top_doc: int = 3,
    top_terms: int = 10,
    use_doc=True,
    context: str = "different things",
):
    specific_terms = specific_terms[:top_terms]
    specific_documents = specific_documents[:top_doc]

    if use_doc:
        PROMPT_TOPICS = ChatPromptTemplate.from_template(promp_template_topics_terms)

        topic_chain = LLMChain(llm=generative_model, prompt=PROMPT_TOPICS)
        clean_topic_name = topic_chain(
            {
                "terms": ", ".join(specific_terms),
                "documents": " \n".join(specific_documents),
                "context": context,
                "language": language,
            }
        )
    else:
        PROMPT_TOPICS_NO_DOCS = ChatPromptTemplate.from_template(
            promp_template_topics_terms_no_docs
        )

        topic_chain = LLMChain(llm=generative_model, prompt=PROMPT_TOPICS_NO_DOCS)
        clean_topic_name = topic_chain(
            {
                "terms": ", ".join(specific_terms),
                "context": context,
                "language": language,
            }
        )

    clean_topic_name = clean_topic_name["text"]

    return clean_topic_name


def get_clean_topic_all(
    generative_model,
    topics: t.List[Topic],
    docs: t.List[Document],
    language: str = "english",
    top_doc: int = 3,
    top_terms: int = 10,
    use_doc=False,
    context: str = "everything",
) -> t.List[Topic]:
    df = get_df_prompt(topics, docs)

    topic_ids = list(df["topic_id"])
    specific_terms = list(df["keywords"])
    top_doc_contents = list(df["content"])

    final_dict = {}
    pbar = tqdm(total=len(topic_ids), desc="Creating new labels for clusters")
    for topic_ic, x, y in zip(topic_ids, specific_terms, top_doc_contents):
        clean_topic_name = get_clean_topic(
            generative_model=generative_model,
            language=language,
            specific_terms=x,
            specific_documents=y,
            use_doc=use_doc,
            top_terms=top_terms,
            top_doc=top_doc,
            context=context,
        )
        final_dict[topic_ic] = clean_topic_name
        pbar.update(1)

    for topic in topics:
        topic.name = final_dict.get(topic.topic_id)

    return topics


def get_df_prompt(topics: t.List[Topic], docs: t.List[Document]) -> pd.DataFrame:
    """
    get a dataframe to input the prompt


    """

    docs_with_ranks = [x for x in docs if x.topic_ranking is not None]

    df_for_prompt = pd.DataFrame(
        {
            "topic_id": [x.topic_ranking.topic_id for x in docs_with_ranks],
            "rank": [x.topic_ranking.rank for x in docs_with_ranks],
            "doc_id": [x.doc_id for x in docs_with_ranks],
        }
    )

    df_for_prompt = df_for_prompt.sort_values(
        ["topic_id", "rank"], ascending=(False, True)
    )
    df_for_prompt = df_for_prompt[["topic_id", "doc_id"]]

    df_doc = pd.DataFrame(
        {
            "doc_id": [x.doc_id for x in docs],
            "content": [x.content for x in docs],
        }
    )

    df_for_prompt = pd.merge(df_for_prompt, df_doc, on="doc_id")
    df_for_prompt = df_for_prompt.groupby("topic_id")["content"].apply(
        lambda x: list(x)
    )

    df_keywords = pd.DataFrame(
        {
            "topic_id": [x.topic_id for x in topics],
            "keywords": [x.name.split(" | ") for x in topics],
        }
    )

    df_for_prompt = pd.merge(df_keywords, df_for_prompt, on="topic_id")

    return df_for_prompt
