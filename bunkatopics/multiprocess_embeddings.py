import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from tqdm import tqdm
import umap

model = SentenceTransformer("distiluse-base-multilingual-cased-v1")


def multi_embed(text):
    emb = model.encode(text, show_progress_bar=False)
    return emb


def get_embeddings(data, index_var, text_var, multiprocessing=True, reduction=5):
    # Embed documents

    data = data[data[text_var].notna()]
    docs = data[text_var].to_list()
    indexes = data[index_var].to_list()

    if multiprocessing:
        with Pool(8) as p:
            embeddings = list(tqdm(p.imap(multi_embed, docs), total=len(docs)))
        df_embeddings = pd.DataFrame(embeddings, index=indexes)
    else:
        embeddings = model.encode(docs, show_progress_bar=True)
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.index = indexes

    if reduction is not None:
        red = umap.UMAP(n_components=reduction, verbose=True, random_state=42)
        red.fit(embeddings)
        embeddings = red.transform(embeddings)
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.index = indexes

    return df_embeddings
