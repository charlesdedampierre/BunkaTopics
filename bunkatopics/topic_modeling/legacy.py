# def rag_query(self, query: str, llm: LLM, top_doc: int = 2) -> BaseRetrievalQA:
#     """
#     Executes a Retrieve-and-Generate (RAG) query using the provided language model and document set.

#     Args:
#         query (str): The query string to be processed.
#         llm: The language model used for generating answers.
#         top_doc (int): The number of top documents to retrieve for the query. Default is 2.

#     Returns:
#         The response from the RAG query, including the answer and source documents.

#     Note:
#         This method utilizes a RetrievalQA chain to answer queries. It retrieves relevant documents
#         based on the query and uses the language model to generate a response. The method is designed
#         to work with complex queries and provide informative answers using the document set.
#     """
#     # Log a message indicating the query is being processed
#     logger.info("Answering your query, please wait a few seconds")

#     # Create a RetrievalQA instance with the specified llm and retriever
#     qa_with_sources_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_doc}),
#         return_source_documents=True,  # Include source documents in the response
#     )

#     # Provide the query to the RetrievalQA instance for answering
#     response = qa_with_sources_chain({"query": query})

#     return response


# characters = string.ascii_letters + string.digits
# random_string = "".join(random.choice(characters) for _ in range(20))

# df_loader = df.copy()
# if metadata is not None:
#     df_loader = df_loader.drop("metadata", axis=1)

# df_loader = pd.DataFrame(sentences, columns=["text"])
# df_loader["doc_id"] = ids

# loader = DataFrameLoader(df_loader, page_content_column="text")
# documents_langchain = loader.load()
# self.vectorstore = Chroma.from_documents(
#     documents_langchain, self.embedding_model, collection_name=random_string
# )

# bunka_ids = [item["doc_id"] for item in self.vectorstore.get()["metadatas"]]
# bunka_docs = self.vectorstore.get()["documents"]
# bunka_embeddings = self.vectorstore._collection.get(include=["embeddings"])[
#     "embeddings"
# ]
