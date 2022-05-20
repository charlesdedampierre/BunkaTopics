import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from tqdm import tqdm
import umap

path = "/Users/charlesdedampierre/Desktop/transformer_models/"
model = SentenceTransformer(path + "distiluse-base-multilingual-cased-v1")


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
        red = umap.UMAP(n_components=reduction, verbose=True)
        red.fit(embeddings)
        embeddings = red.transform(embeddings)
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.index = indexes

    return df_embeddings


if __name__ == "__main__":

    path = "/Users/charlesdedampierre/Desktop/ENS Projects/humility"
    df_index = pd.read_csv(path + "/extended_dataset/extended_training_dataset.csv")
    df_index = df_index[["body", "label"]].drop_duplicates()
    df_index = df_index.sample(100)
    df_index["index"] = df_index.index

    df_embeddings = get_embeddings(
        df_index, index_var="index", text_var="body", multiprocessing=False, reduction=5
    )

    print(df_embeddings)
