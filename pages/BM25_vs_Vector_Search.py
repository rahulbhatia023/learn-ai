import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from rank_bm25 import BM25Okapi

navigation_page_key = "bm25vs_navigation_page"
if navigation_page_key not in st.session_state:
    st.session_state[navigation_page_key] = 1

file_uploader_key = "bm25vs_file_uploader"

uploaded_file_key = "bm25vs_uploaded_file"
if uploaded_file_key not in st.session_state:
    st.session_state[uploaded_file_key] = None

loaded_pdf_pages_key = "bm25vs_pdf_pages"
if loaded_pdf_pages_key not in st.session_state:
    st.session_state[loaded_pdf_pages_key] = None

chunk_size_key = "bm25vs_chunk_size"
if chunk_size_key not in st.session_state:
    st.session_state[chunk_size_key] = None

chunk_overlap_key = "bm25vs_chunk_overlap"
if chunk_overlap_key not in st.session_state:
    st.session_state[chunk_overlap_key] = None

chunks_key = "bm25vs_chunks"
if chunks_key not in st.session_state:
    st.session_state[chunks_key] = []

user_query_key = "bm25vs_user_query"
if user_query_key not in st.session_state:
    st.session_state[user_query_key] = ""

bm25_similar_documents_key = "bm25vs_bm25_documents"
if bm25_similar_documents_key not in st.session_state:
    st.session_state[bm25_similar_documents_key] = None

vs_similar_documents_key = "bm25vs_vs_documents"
if vs_similar_documents_key not in st.session_state:
    st.session_state[vs_similar_documents_key] = None

st.set_page_config(
    page_title="BM25 vs Vector Search",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
    layout="wide",
)

st.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        .stApp {
            font-family: 'Poppins';
            background-color: #16423C;
        }

        [title="Show password text"] {
            display: none;
        }
    </style>
    """
)

st.html(
    f"<h1 style='font-family:Poppins; color:#C4DAD2; text-align: center'>Best Matching 25 (BM25) vs Vector Search</h1><br/>"
)

# PAGE-1: Upload PDF

if st.session_state[navigation_page_key] == 1:
    _, col2, _ = st.columns([1, 6, 1])

    with col2:
        with st.container(border=True):
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Upload Document</h3>"
            )

            if uploaded_file := st.file_uploader(
                label="Upload PDF File",
                label_visibility="hidden",
                type=["pdf"],
                key=file_uploader_key,
            ):
                if (
                    not st.session_state[uploaded_file_key]
                    or st.session_state[uploaded_file_key] != uploaded_file
                ):
                    st.session_state[uploaded_file_key] = uploaded_file

                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(st.session_state[uploaded_file_key].getvalue())

                    pdf_loader = PyPDFLoader(temp_file.name)
                    pdf_pages = pdf_loader.load()

                    st.session_state[loaded_pdf_pages_key] = pdf_pages

            if st.session_state[uploaded_file_key]:
                st.html("<br/>")
                st.success(
                    body=f"Uploaded file: {st.session_state[uploaded_file_key].name}, Total pages: {len(st.session_state[loaded_pdf_pages_key])}",
                    icon="✅",
                )

# PAGE-2: Create Chunks

elif st.session_state[navigation_page_key] == 2:
    _, col2, _ = st.columns([1, 6, 1])

    with col2:
        with st.container(border=True):
            if st.session_state[uploaded_file_key]:
                st.html(
                    "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Split document into chunks</h3>"
                )

                _, col2, _ = st.columns([1, 2, 1])
                with col2:
                    if chunk_size := st.number_input(label="Chunk size", value=4000):
                        st.session_state[chunk_size_key] = chunk_size

                    if chunk_overlap := st.number_input(
                        label="Chunk overlap", value=200
                    ):
                        st.session_state[chunk_overlap_key] = chunk_overlap

                    st.html("<br/>")

                _, col2, _ = st.columns([1, 0.5, 1])
                with col2:
                    if st.button(
                        label="Create chunks",
                        type="secondary",
                        use_container_width=True,
                    ):
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=st.session_state[chunk_size_key],
                            chunk_overlap=st.session_state[chunk_overlap_key],
                        )

                        chunks = text_splitter.split_documents(
                            st.session_state[loaded_pdf_pages_key]
                        )

                        st.session_state[chunks_key] = []
                        for chunk_id, chunk_obj in enumerate(chunks):
                            st.session_state[chunks_key].append(
                                {"chunk_id": chunk_id + 1, "chunk": chunk_obj}
                            )

            st.html("<br/>")

            if st.session_state[chunks_key]:
                with st.container(border=True):
                    st.html(
                        "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunks</p>"
                    )

                    for chunk_info in st.session_state[chunks_key]:
                        with st.expander(f"CHUNK: {chunk_info["chunk_id"]}"):
                            st.text(chunk_info["chunk"].page_content)

# PAGE-3: Similarity Search

elif st.session_state[navigation_page_key] == 3:
    if st.session_state[chunks_key]:
        with st.container(border=True):
            _, col2, _ = st.columns([1, 4, 1])

            with col2:
                st.html(
                    "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Similarity Search</h3>"
                )

                if user_query := st.text_input(
                    label="User Query", value=st.session_state[user_query_key]
                ):
                    st.session_state[user_query_key] = user_query

                st.html("<br/>")

                _, col2, _ = st.columns([1, 0.3, 1])

                with col2:
                    if st.button(
                        label="Search",
                        type="secondary",
                        use_container_width=True,
                    ):
                        if not st.session_state[user_query_key]:
                            st.error("Please enter user query.", icon="🚨")
                        else:
                            chunks = st.session_state[chunks_key]

                            # BM25

                            tokenized_query = st.session_state[user_query_key].split()

                            chunks_content = [
                                item["chunk"].page_content for item in chunks
                            ]

                            tokenized_chunks_content = [
                                chunk_content.split()
                                for chunk_content in chunks_content
                            ]

                            chunks_with_bm25_scores = list(
                                zip(
                                    chunks,
                                    BM25Okapi(tokenized_chunks_content).get_scores(
                                        query=tokenized_query
                                    ),
                                )
                            )

                            bm25_retriever = BM25Retriever.from_documents(
                                documents=[item["chunk"] for item in chunks]
                            )

                            bm25_similar_documents = bm25_retriever.invoke(
                                input=st.session_state[user_query_key]
                            )

                            bm25_similar_documents_with_scores = []
                            for bm25_similar_document in bm25_similar_documents:
                                for chunk_with_bm25_score in chunks_with_bm25_scores:
                                    if (
                                        bm25_similar_document.page_content
                                        == chunk_with_bm25_score[0][
                                            "chunk"
                                        ].page_content
                                    ):
                                        bm25_similar_documents_with_scores.append(
                                            chunk_with_bm25_score
                                        )

                            st.session_state[bm25_similar_documents_key] = sorted(
                                bm25_similar_documents_with_scores,
                                key=lambda x: x[1],
                                reverse=True,
                            )

                            # vector search

                            vector_store = FAISS.from_documents(
                                documents=[item["chunk"] for item in chunks],
                                embedding=OpenAIEmbeddings(
                                    openai_api_key=SecretStr(
                                        os.environ["OPENAI_API_KEY"]
                                    )
                                ),
                            )

                            chunks_with_vector_search_scores = (
                                vector_store.similarity_search_with_score(
                                    query=st.session_state[user_query_key]
                                )
                            )

                            vector_store_retriever = vector_store.as_retriever()

                            vector_store_similar_documents = (
                                vector_store_retriever.invoke(
                                    input=st.session_state[user_query_key]
                                )
                            )

                            vector_store_similar_documents_with_scores = []
                            for (
                                vector_store_similar_document
                            ) in vector_store_similar_documents:
                                for (
                                    chunk_with_vector_search_score
                                ) in chunks_with_vector_search_scores:
                                    if (
                                        vector_store_similar_document.page_content
                                        == chunk_with_vector_search_score[
                                            0
                                        ].page_content
                                    ):
                                        for chunk_info in chunks:
                                            if (
                                                chunk_info["chunk"].page_content
                                                == chunk_with_vector_search_score[
                                                    0
                                                ].page_content
                                            ):
                                                chunk_info["chunk_score"] = (
                                                    chunk_with_vector_search_score[1]
                                                )

                                                vector_store_similar_documents_with_scores.append(
                                                    chunk_info
                                                )

                            st.session_state[vs_similar_documents_key] = sorted(
                                vector_store_similar_documents_with_scores,
                                key=lambda x: x["chunk_score"],
                                reverse=True,
                            )

            if (
                st.session_state[bm25_similar_documents_key]
                and st.session_state[vs_similar_documents_key]
            ):
                st.html("<br/>")

                bm25, vector_search = st.columns([1, 1])

                with bm25:
                    with st.container(border=True):
                        st.html(
                            "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>BM25</p>"
                        )

                        for bm25_similar_document in st.session_state[
                            bm25_similar_documents_key
                        ]:
                            with st.expander(
                                f"CHUNK: {bm25_similar_document[0]["chunk_id"]}, SCORE: {bm25_similar_document[1]}"
                            ):
                                st.text(bm25_similar_document[0]["chunk"].page_content)

                with vector_search:
                    with st.container(border=True):
                        st.html(
                            "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>Vector Search</p>"
                        )

                        for vs_similar_document in st.session_state[
                            vs_similar_documents_key
                        ]:
                            with st.expander(
                                f"CHUNK: {vs_similar_document["chunk_id"]}, SCORE: {vs_similar_document["chunk_score"]}"
                            ):
                                st.text(vs_similar_document["chunk"].page_content)

_, mid, _ = st.columns([1, 3, 1])
with mid:
    previous_col, _, next_col = st.columns([1, 1, 1])

    previous_button_disabled = False
    next_button_disabled = False

    if st.session_state[navigation_page_key] == 1:
        previous_button_disabled = True
        if not st.session_state[uploaded_file_key]:
            next_button_disabled = True
    elif st.session_state[navigation_page_key] == 2:
        if not st.session_state[chunks_key]:
            next_button_disabled = True

    with previous_col:
        st.html("<br/>")
        if st.button(
            label="Previous",
            type="primary",
            use_container_width=True,
            disabled=previous_button_disabled,
        ):
            st.session_state[navigation_page_key] -= 1
            st.rerun()

    with next_col:
        st.html("<br/>")
        if st.button(
            label="Next",
            type="primary",
            use_container_width=True,
            disabled=next_button_disabled,
        ):
            st.session_state[navigation_page_key] += 1
            st.rerun()
