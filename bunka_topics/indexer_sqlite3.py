import sqlite3
import pandas as pd
import os


def indexer(docs: list, terms: list, db_path="."):

    terms = [x for x in terms if not x.startswith('"')]
    terms = [x for x in terms if not x.endswith('"')]
    terms = [x for x in terms if not x.startswith("'")]
    terms = [x for x in terms if not x.endswith("'")]
    terms = [x.replace('"', "") for x in terms]
    terms = ['"' + x + '"' for x in terms]

    conn = sqlite3.connect(db_path + "/database.db")

    # insert the docs
    df_docs = pd.DataFrame(docs, columns=["data"])
    df_docs["neo_id"] = df_docs.index
    df_docs.to_sql("docs", conn, if_exists="replace", index=False)

    # insert the terms
    df_terms = pd.DataFrame(terms, columns=["words"])
    df_terms.to_sql("terms", conn, if_exists="replace", index=False)

    # conn.enable_load_extension(True)
    c = conn.cursor()

    # In case anything exists already
    c.execute("DROP table IF EXISTS abstractsearch;")
    c.execute("DROP table IF EXISTS uniqueterms;")
    c.execute("DROP table IF EXISTS indexed_terms;")
    conn.commit()

    # Starts the FTS5 abstract Search
    c.execute("CREATE VIRTUAL TABLE abstractsearch USING fts5(neo_id, data);")
    conn.commit()

    # Insert the documents to serach terms from
    c.execute(
        "INSERT INTO abstractsearch SELECT neo_id AS neo_id, data AS data FROM docs;"
    )
    conn.commit()

    # Create a unique terms table based on the terms table
    c.execute("CREATE TABLE uniqueterms AS SELECT DISTINCT words AS words FROM terms;")
    conn.commit()

    # Create the output tabke with the result of the TFS5 Search
    c.execute(
        "CREATE TABLE indexed_terms AS SELECT neo_id, words FROM abstractsearch, uniqueterms WHERE abstractsearch.data MATCH uniqueterms.words COLLATE NOCASE;"
        ""
    )

    conn.commit()

    c.execute("DROP table IF EXISTS abstractsearch;")
    c.execute("DROP table IF EXISTS uniqueterms;")

    df_docs_table = pd.read_sql_query("SELECT * FROM docs", conn)
    df_docs_table = df_docs_table.rename(columns={"data": "docs"})

    df_indexed_table = pd.read_sql_query("SELECT * FROM indexed_terms", conn)
    final = pd.merge(df_docs_table, df_indexed_table, on="neo_id")
    final = final[["docs", "words"]].copy()

    final["words"] = final["words"].apply(lambda x: x.strip('"'))

    os.remove(db_path + "/database.db")

    return final
