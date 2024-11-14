from typing import Union, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from pydantic import SecretStr


class DocumentsRetrieverTool(BaseTool):
    pdf_file: str = Field(..., description="Uploaded PDF file")
    openai_api_key: str = Field(..., description="OpenAI API key")

    name: str = "documents-retriever"
    description: str = "Retrieve documents chunks"

    def _run(self, query: str) -> Union[Dict, str]:
        return "\n\n".join(
            doc.page_content
            for doc in FAISS.from_documents(
                RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                ).split_documents(PyPDFLoader(self.pdf_file).load()),
                OpenAIEmbeddings(openai_api_key=SecretStr(self.openai_api_key)),
            )
            .as_retriever()
            .invoke(query)
        )
