import os
import tempfile
import time

import numpy as np
import streamlit as st
from langchain.prompts import Prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

chunks_with_bm25_scores_key = "bm25vs_chunks_with_bm25_scores"
if chunks_with_bm25_scores_key not in st.session_state:
    st.session_state[chunks_with_bm25_scores_key] = None

chunks_with_vector_scores_key = "bm25vs_chunks_with_vector_scores"
if chunks_with_vector_scores_key not in st.session_state:
    st.session_state[chunks_with_vector_scores_key] = None

chunks_with_rrf_scores_key = "bm25vs_chunks_with_rrf_scores"
if chunks_with_rrf_scores_key not in st.session_state:
    st.session_state[chunks_with_rrf_scores_key] = None

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
                    "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Similarity Search Scores</h3>"
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

                            bm25_scores = BM25Okapi(
                                tokenized_chunks_content
                            ).get_scores(query=tokenized_query)

                            bm25_ranks = (
                                len(bm25_scores) - bm25_scores.argsort().argsort()
                            )

                            chunks_with_bm25_scores = list(
                                zip(chunks, bm25_scores, bm25_ranks)
                            )

                            st.session_state[chunks_with_bm25_scores_key] = (
                                chunks_with_bm25_scores
                            )

                            # Vector Search

                            vector_store = FAISS.from_documents(
                                documents=[item["chunk"] for item in chunks],
                                embedding=OpenAIEmbeddings(
                                    openai_api_key=SecretStr(
                                        os.environ["OPENAI_API_KEY"]
                                    ),
                                    model="text-embedding-3-large",
                                ),
                            )

                            vector_scores = vector_store.similarity_search_with_score(
                                query=st.session_state[user_query_key],
                                k=len(chunks),
                            )

                            vector_ranks = (
                                len(vector_scores)
                                - np.array(list(map(lambda tup: tup[1], vector_scores)))
                                .argsort()
                                .argsort()
                            )

                            vector_scores = zip(vector_scores, vector_ranks)

                            chunks_with_vector_scores = []
                            for chunk_with_vector_score in vector_scores:
                                for chunk in chunks:
                                    if (
                                        chunk["chunk"].page_content
                                        == chunk_with_vector_score[0][0].page_content
                                    ):
                                        chunks_with_vector_scores.append(
                                            (
                                                chunk,
                                                chunk_with_vector_score[0][1],
                                                chunk_with_vector_score[1],
                                            )
                                        )

                            st.session_state[chunks_with_vector_scores_key] = sorted(
                                chunks_with_vector_scores,
                                key=lambda x: x[0]["chunk_id"],
                            )

            if (
                st.session_state[chunks_with_bm25_scores_key]
                and st.session_state[chunks_with_vector_scores_key]
            ):
                st.html("<br/>")

                bm25, vector_search = st.columns([1, 1])

                with bm25:
                    with st.container(border=True):
                        st.html(
                            "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>BM25</h3>"
                        )

                        for chunk_with_bm25_score in st.session_state[
                            chunks_with_bm25_scores_key
                        ]:
                            with st.expander(
                                f"CHUNK: {chunk_with_bm25_score[0]["chunk_id"]}, SCORE: {chunk_with_bm25_score[1]}, RANK: {chunk_with_bm25_score[2]}"
                            ):
                                st.text(chunk_with_bm25_score[0]["chunk"].page_content)

                with vector_search:
                    with st.container(border=True):
                        st.html(
                            "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Vector Search</h3>"
                        )

                        for chunk_with_vector_score in st.session_state[
                            chunks_with_vector_scores_key
                        ]:
                            with st.expander(
                                f"CHUNK: {chunk_with_vector_score[0]["chunk_id"]}, SCORE: {chunk_with_vector_score[1]}, RANK: {chunk_with_vector_score[2]}"
                            ):
                                st.text(
                                    chunk_with_vector_score[0]["chunk"].page_content
                                )

# PAGE-4: Re-Ranking

elif st.session_state[navigation_page_key] == 4:
    if (
        st.session_state[chunks_with_bm25_scores_key]
        and st.session_state[chunks_with_vector_scores_key]
    ):
        with st.container(border=True):
            _, col2, _ = st.columns([1, 4, 1])

            with col2:
                st.html(
                    "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Reciprocal Rank Fusion (RRF)</h3>"
                )

                st.html("<br/>")

                def rrf_score(keyword_rank: int, semantic_rank: int) -> float:
                    k = 60
                    return 1 / (k + keyword_rank) + 1 / (k + semantic_rank)

                rrf_scores = [
                    (chunk, rrf_score(bm25_rank, vector_rank))
                    for (chunk, _, bm25_rank), (_, _, vector_rank) in zip(
                        st.session_state[chunks_with_bm25_scores_key],
                        st.session_state[chunks_with_vector_scores_key],
                    )
                ]

                rrf_ranks = (
                    len(rrf_scores)
                    - np.array(list(map(lambda tup: tup[1], rrf_scores)))
                    .argsort()
                    .argsort()
                )

                chunks_with_rrf_scores = zip(rrf_scores, rrf_ranks)

                st.session_state[chunks_with_rrf_scores_key] = sorted(
                    [
                        (
                            chunk,
                            rrf_score,
                            rrf_rank,
                        )
                        for (
                            chunk,
                            rrf_score,
                        ), rrf_rank in chunks_with_rrf_scores
                    ],
                    key=lambda x: x[0]["chunk_id"],
                )

                for chunk_with_rrf_score in st.session_state[
                    chunks_with_rrf_scores_key
                ]:
                    with st.expander(
                        f"CHUNK: {chunk_with_rrf_score[0]["chunk_id"]}, SCORE: {chunk_with_rrf_score[1]}, RANK: {chunk_with_rrf_score[2]}"
                    ):
                        st.text(chunk_with_rrf_score[0]["chunk"].page_content)

# PAGE-5: Generate Response

elif st.session_state[navigation_page_key] == 5:
    if st.session_state[chunks_with_rrf_scores_key]:
        st.html(
            "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Generate Response</h3>"
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.html(
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>LLM Input</p>"
                )

                with st.container(border=True):
                    st.html(
                        "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>User Query</p>"
                    )
                    st.write(st.session_state[user_query_key])

                with st.container(border=True):
                    st.html(
                        "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>LLM Context</p>"
                    )

                    top_4_chunks = sorted(
                        st.session_state[chunks_with_rrf_scores_key], key=lambda x: x[2]
                    )[0:4]

                    for chunk, rrf_score, rrf_rank in top_4_chunks:
                        with st.expander(
                            f"CHUNK: {chunk["chunk_id"]}, SCORE: {rrf_score}, RANK: {rrf_rank}"
                        ):
                            st.text(chunk["chunk"].page_content)

        with col2:
            with st.container(border=True):
                st.html(
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>LLM Output</p>"
                )

                response = None

                col1, col2, col3 = st.columns([1, 3, 1])

                with col2:
                    if st.button(
                        label="Generate Response",
                        type="secondary",
                        use_container_width=True,
                    ):
                        llm = ChatOpenAI(
                            model_name="gpt-4o",
                            openai_api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                        )

                        prompt = Prompt.from_template(
                            """
                            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                            Question: {question} 

                            Context: {context} 

                            Answer:
                            """
                        )

                        rag_chain = prompt | llm

                        llm_context = "\n\n".join(
                            chunk["chunk"].page_content for chunk, _, _ in top_4_chunks
                        )

                        response = rag_chain.invoke(
                            {
                                "context": llm_context,
                                "question": st.session_state[user_query_key],
                            }
                        ).content

                def stream_data():
                    for word in response.split(" "):
                        yield word + " "
                        time.sleep(0.04)

                if response:
                    with st.container(border=True):
                        st.write_stream(stream_data)
                        st.balloons()

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
    elif st.session_state[navigation_page_key] == 3:
        if (
            not st.session_state[chunks_with_bm25_scores_key]
            or not st.session_state[chunks_with_vector_scores_key]
        ):
            next_button_disabled = True
    elif st.session_state[navigation_page_key] == 4:
        if not st.session_state[chunks_with_rrf_scores_key]:
            next_button_disabled = True
    elif st.session_state[navigation_page_key] == 5:
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
