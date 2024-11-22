import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    st.session_state[chunks_key] = None


def create_chunks():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state[chunk_size_key],
        chunk_overlap=st.session_state[chunk_overlap_key],
    )

    chunks = text_splitter.split_documents(st.session_state[pdf_pages_key])

    st.session_state["dc_chunks"] = chunks


st.set_page_config(
    page_title="Document Chunking",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

st.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        .stApp {
            font-family: 'Poppins';
            background-color: #16423C;
        }
    </style>
    """
)

st.html(f"<h2 style='font-family:Poppins; color:#C4DAD2';>Document Chunking</h2><br/>")

with st.sidebar:
    st.html(
        "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Upload PDF File</h3>"
    )

    st.file_uploader(
        label="Upload PDF File",
        label_visibility="hidden",
        type=["pdf"],
        key=file_uploader_key,
    )

if st.session_state[file_uploader_key] and not st.session_state[uploaded_file_key]:
    st.session_state[uploaded_file_key] = st.session_state[file_uploader_key]

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(st.session_state[uploaded_file_key].getvalue())

    pdf_loader = PyPDFLoader(temp_file.name)
    pdf_pages = pdf_loader.load()

    st.session_state["dc_pdf_pages"] = pdf_pages

if st.session_state[uploaded_file_key]:
    st.success(
        body=f"Uploaded file: {st.session_state[uploaded_file_key].name}, Total pages: {len(st.session_state["dc_pdf_pages"])}",
        icon="✅",
    )

    with st.sidebar:
        st.divider()

        st.html(
            "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunking Options</h3>"
        )

        if chunk_size := st.number_input(label="Chunk size", value=4000):
            st.session_state[chunk_size_key] = chunk_size

        if chunk_overlap := st.number_input(label="Chunk overlap", value=200):
            st.session_state[chunk_overlap_key] = chunk_overlap

        st.html("<br/>")

        if st.button(label="Create chunks", type="primary", use_container_width=True):
            create_chunks()

    st.html("<br/>")

    if st.session_state[chunks_key]:
        with st.container(border=True):
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunks</h3>"
            )

            for idx, chunk in enumerate(st.session_state[chunks_key]):
                with st.expander(f"Chunk: {idx + 1}"):
                    st.write(chunk.page_content)
