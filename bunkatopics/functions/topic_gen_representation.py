import pandas as pd
import openai
from tqdm import tqdm
from bunkatopics.datamodel import Topic, Document
import typing as t


# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that is described the following keywords separated by ||: 
'[KEYWORDS]'
/n

Here are some examples of documents separated by || in the topic:
[DOCUMENTS]

/n

Based on the information about the topic above, please create a short label of this topic. The keywords are the most important elements. The documents are here to
help desimbiguate topics and give the context in which the keywords are expressed.

The documents are just examples, do not be too specific in chosing the topic.
Make sure you to only return the label and nothing more.

[/INST]

"""


def get_clean_topics(
    df_for_prompt: pd.DataFrame, topics: t.List[Topic], openai_key: str
) -> t.List[Topic]:
    openai.api_key = openai_key
    keywords_list = list(df_for_prompt["keywords"])

    topic_id_list = list(df_for_prompt["topic_id"])
    content_list = list(df_for_prompt["content"])

    final = []

    pbar = tqdm(total=len(topic_id_list), desc="Giving Names to topics using OpenAI...")
    for topic_id, keywords, contents in zip(topic_id_list, keywords_list, content_list):
        try:
            contents = " || ".join(contents)
            keywords = " || ".join(keywords)
            new_prompt = main_prompt.replace("[DOCUMENTS]", contents).replace(
                "[KEYWORDS]", keywords
            )
            res = get_results_from_gpt(new_prompt)
            final.append({"topic_id": topic_id, "topic_gen_name": res})
            pbar.update(1)
        except:
            final.append({"topic_id": topic_id, "topic_gen_name": "ERROR"})

    df_final = pd.DataFrame(final)

    topic_ids = list(df_final["topic_id"])
    names = list(df_final["topic_gen_name"])
    dict_topic_gen_name = {x: y for x, y in zip(topic_ids, names)}

    for topic in topics:
        topic.name = dict_topic_gen_name.get(topic.topic_id, [])

    return topics


def get_results_from_gpt(prompt):
    model_type = "gpt-3.5-turbo"
    model_type = "gpt-4"
    completion = openai.ChatCompletion.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    res = completion.choices[0].message["content"]

    return res


def get_df_prompt(topics: t.List[Topic], docs: t.List[Document]) -> pd.DataFrame:
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
