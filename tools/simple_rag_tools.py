from typing import Union, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from pydantic import Field


class DocumentsRetrieverTool(BaseTool):
    faiss_vector_store: FAISS = Field(..., description="FAISS vector store")

    name: str = "documents-retriever"
    description: str = "Retrieve similar documents chunks"

    def _run(self, query: str) -> Union[Dict, str]:
        return "\n\n".join(
            doc.page_content
            for doc in self.faiss_vector_store.as_retriever().invoke(query)
        )
