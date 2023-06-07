import pandas as pd
from ..datamodel import Document
import typing as t
from sklearn.metrics.pairwise import cosine_similarity


def vector_search(
    docs: t.List[Document], model_hf, user_input: str = "I love you"
) -> pd.DataFrame:
    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_emb = df_docs[["doc_id", "embedding"]]
    df_emb = df_emb.set_index("doc_id")
    df_emb = pd.DataFrame(list(df_emb["embedding"]))
    df_emb.index = df_docs["doc_id"]

    query_embedding = model_hf.embed_documents([user_input])

    df_query = pd.DataFrame(query_embedding)
    df_query.index = ["query_id"]

    # Compute the Cosine Similarity
    full_emb = pd.concat([df_emb, df_query])
    df_bert = pd.DataFrame(cosine_similarity(full_emb))

    df_bert.index = full_emb.index
    df_bert.columns = full_emb.index

    df_bert = df_bert.iloc[-1:,].T
    df_bert = df_bert.sort_values("query_id", ascending=False).reset_index()
    df_bert = df_bert[1:]

    df_bert = df_bert.rename(columns={"index": "doc_id"})
    final_df = pd.merge(df_bert, df_docs[["doc_id", "content"]], on="doc_id")
    final_df = final_df.rename(columns={"query_id": "cosine_similarity_score"})

    return final_df
