from typing import Dict, List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain_core.documents import Document

class ReduncdantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str):
        # calculate embedding for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embedding and feed them into that

        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self, query: str):
        return []