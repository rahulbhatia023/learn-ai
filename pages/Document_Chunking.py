import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

file_uploader_key = "dc_file_uploader"

uploaded_file_key = "dc_uploaded_file"
if uploaded_file_key not in st.session_state:
    st.session_state[uploaded_file_key] = None

pdf_pages_key = "dc_pdf_pages"
if pdf_pages_key not in st.session_state:
    st.session_state[pdf_pages_key] = None

chunk_size_key = "dc_chunk_size"
if chunk_size_key not in st.session_state:
    st.session_state[chunk_size_key] = None

chunk_overlap_key = "dc_chunk_overlap"
if chunk_overlap_key not in st.session_state:
    st.session_state[chunk_overlap_key] = None

chunks_key = "dc_chunks"
if chunks_key not in st.session_state:
    st.session_state[chunks_key] = []

chunks_with_embeddings_key = "dc_chunks_with_embeddings"
if chunks_with_embeddings_key not in st.session_state:
    st.session_state[chunks_with_embeddings_key] = []

navigation_page_key = "dc_navigation_page"
if navigation_page_key not in st.session_state:
    st.session_state[navigation_page_key] = 1

openai_api_key = "OPENAI_API_KEY"
if openai_api_key not in st.session_state:
    st.session_state[openai_api_key] = ""

st.set_page_config(
    page_title="Document Chunking",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
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

st.html(f"<h2 style='font-family:Poppins; color:#C4DAD2; text-align: center'>Document Chunking</h2><br/>")

with st.container():
    # PAGE-1: Upload PDF

    if st.session_state[navigation_page_key] == 1:
        st.html(
            "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Upload PDF File</h3>"
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

                st.session_state["dc_pdf_pages"] = pdf_pages

        if st.session_state[uploaded_file_key]:
            st.html("<br/>")
            st.success(
                body=f"Uploaded file: {st.session_state[uploaded_file_key].name}, Total pages: {len(st.session_state["dc_pdf_pages"])}",
                icon="✅",
            )

    # PAGE-2: Create Chunks

    elif st.session_state[navigation_page_key] == 2:
        if st.session_state[uploaded_file_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunking Options</h3>"
            )

            if chunk_size := st.number_input(label="Chunk size", value=4000):
                st.session_state[chunk_size_key] = chunk_size

            if chunk_overlap := st.number_input(label="Chunk overlap", value=200):
                st.session_state[chunk_overlap_key] = chunk_overlap

            st.html("<br/>")

            _, col2, _ = st.columns([1, 1, 1])
            with col2:
                if st.button(label="Create chunks", type="secondary", use_container_width=True):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=st.session_state[chunk_size_key],
                        chunk_overlap=st.session_state[chunk_overlap_key],
                    )

                    chunks = text_splitter.split_documents(st.session_state[pdf_pages_key])

                    for chunk_id, chunk_obj in enumerate(chunks):
                        st.session_state[chunks_key].append(
                            {"chunk_id": chunk_id + 1, "chunk": chunk_obj}
                        )

        st.html("<br/>")

        if st.session_state[chunks_key]:
            with st.container(border=True):
                st.html(
                    "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunks</h3>"
                )

                for chunk_info in st.session_state[chunks_key]:
                    with st.expander(f"Chunk: {chunk_info["chunk_id"]}"):
                        st.text(chunk_info["chunk"].page_content)

    # PAGE-3: Create embeddings

    elif st.session_state[navigation_page_key] == 3:
        if st.session_state[chunks_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Create Embeddings</h3>"
            )

            if not st.session_state[openai_api_key]:
                if openai_api_key in st.secrets and st.secrets[openai_api_key]:
                    st.session_state[openai_api_key] = st.secrets[openai_api_key]
                else:
                    st.error("Please enter your OpenAI API key.", icon="🚨")

            if api_key := st.text_input(
                    label=f"{openai_api_key}", value=st.session_state[openai_api_key], type="password"
            ):
                st.session_state[openai_api_key] = api_key

            if st.session_state[openai_api_key]:
                st.html("<br/>")
                _, col2, _ = st.columns([1, 1, 1])
                with col2:
                    if st.button(
                            label="Create embeddings",
                            type="secondary",
                            use_container_width=True,
                    ):
                        embeddings = OpenAIEmbeddings(
                            openai_api_key=SecretStr(st.session_state[openai_api_key]),
                        )

                        for chunk_info in st.session_state[chunks_key]:
                            chunk_info["chunk_embedding"] = embeddings.embed_query(
                                text=chunk_info["chunk"].page_content
                            )
                            st.session_state[chunks_with_embeddings_key].append(chunk_info)

        if st.session_state[chunks_with_embeddings_key]:
            st.html("<br/>")

            with st.container(border=True):
                st.html(
                    "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunks with embeddings</h3>"
                )

                for chunk_info in st.session_state[chunks_key]:
                    with st.expander(f"Chunk: {chunk_info["chunk_id"]}"):
                        chunk_content, chunk_embedding = st.tabs(["Chunk Content", "Chunk Embedding"])
                        chunk_content.text(chunk_info["chunk"].page_content)
                        chunk_embedding.text(chunk_info["chunk_embedding"])

container = st.container()
previous_col, _, next_col = container.columns([1, 1, 1])

previous_button_disabled = False
next_button_disabled = False

if st.session_state[navigation_page_key] == 1:
    if not st.session_state[uploaded_file_key]:
        previous_button_disabled = True
        next_button_disabled = True
    else:
        previous_button_disabled = True
elif st.session_state[navigation_page_key] == 2:
    if not st.session_state[chunks_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 3:
    if not st.session_state[chunks_with_embeddings_key]:
        next_button_disabled = True

with previous_col:
    st.html("<br/>")
    if st.button(
            label="Previous",
            type="primary",
            use_container_width=True,
            disabled=previous_button_disabled
    ):
        st.session_state[navigation_page_key] -= 1
        st.rerun()

with next_col:
    st.html("<br/>")
    if st.button(
            label="Next",
            type="primary",
            use_container_width=True,
            disabled=next_button_disabled
    ):
        st.session_state[navigation_page_key] += 1
        st.rerun()