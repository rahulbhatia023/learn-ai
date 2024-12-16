import tempfile
import time

import numpy as np
from langchain.prompts import Prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse, ResultType
from pydantic import SecretStr, BaseModel
from rank_bm25 import BM25Okapi

from common.theme import *

navigation_page_key = "cr_navigation_page"
if navigation_page_key not in st.session_state:
    st.session_state[navigation_page_key] = 1

uploaded_file_key = "cr_uploaded_file"
if uploaded_file_key not in st.session_state:
    st.session_state[uploaded_file_key] = None

document_file_path_key = "cr_document_file_path"
if document_file_path_key not in st.session_state:
    st.session_state[document_file_path_key] = None

loaded_pdf_pages_key = "cr_pdf_pages"
if loaded_pdf_pages_key not in st.session_state:
    st.session_state[loaded_pdf_pages_key] = None

chunk_size_key = "cr_chunk_size"
if chunk_size_key not in st.session_state:
    st.session_state[chunk_size_key] = None

chunk_overlap_key = "cr_chunk_overlap"
if chunk_overlap_key not in st.session_state:
    st.session_state[chunk_overlap_key] = None

chunks_key = "cr_chunks"
if chunks_key not in st.session_state:
    st.session_state[chunks_key] = []

chunks_with_context_key = "cr_chunks_with_context"
if chunks_with_context_key not in st.session_state:
    st.session_state[chunks_with_context_key] = []

user_query_key = "cr_user_query"
if user_query_key not in st.session_state:
    st.session_state[user_query_key] = ""

chunks_with_bm25_scores_key = "cr_chunks_with_bm25_scores"
if chunks_with_bm25_scores_key not in st.session_state:
    st.session_state[chunks_with_bm25_scores_key] = []

chunks_with_vector_scores_key = "cr_chunks_with_vector_scores"
if chunks_with_vector_scores_key not in st.session_state:
    st.session_state[chunks_with_vector_scores_key] = []

chunks_with_rrf_scores_key = "cr_chunks_with_rrf_scores"
if chunks_with_rrf_scores_key not in st.session_state:
    st.session_state[chunks_with_rrf_scores_key] = []

openai_api_key = "OPENAI_API_KEY"
if openai_api_key not in st.session_state:
    st.session_state[openai_api_key] = ""

llama_cloud_api_key = "LLAMA_CLOUD_API_KEY"
if llama_cloud_api_key not in st.session_state:
    st.session_state[llama_cloud_api_key] = ""


class Chunk(BaseModel):
    id: int
    content: str
    metadata: dict
    context: str = None
    contextualized_content: str = None
    bm25_score: float = None
    bm25_rank: int = None
    vector_score: float = None
    vector_rank: int = None
    rrf_score: float = None
    rrf_rank: int = None


st.set_page_config(
    page_title="Contextual Retrieval RAG",
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

st.html(f"<h1 style={app_title_style}>Contextual Retrieval RAG</h1><br/>")

# PAGE-1: Upload Document

if st.session_state[navigation_page_key] == 1:
    with container(key="upload_document_container"):
        container_title("Upload Document")

        _, col2, _ = st.columns([1, 5, 1])
        with col2:
            if uploaded_file := st.file_uploader(
                label="Upload PDF File", label_visibility="hidden", type=["pdf"]
            ):
                if (
                    not st.session_state[uploaded_file_key]
                    or st.session_state[uploaded_file_key] != uploaded_file
                ):
                    st.session_state[uploaded_file_key] = uploaded_file

                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.write(st.session_state[uploaded_file_key].getvalue())

                    st.session_state[document_file_path_key] = temp_file.name

                    pdf_loader = PyPDFLoader(temp_file.name)
                    pdf_pages = pdf_loader.load()

                    st.session_state[loaded_pdf_pages_key] = pdf_pages

            if st.session_state[uploaded_file_key]:
                st.success(
                    body=f"Uploaded file: {st.session_state[uploaded_file_key].name}, Total pages: {len(st.session_state[loaded_pdf_pages_key])}",
                    icon="✅",
                )

# PAGE-2: Split document into chunks

elif st.session_state[navigation_page_key] == 2:
    if st.session_state[uploaded_file_key]:
        with container(key="create_chunks_container"):
            container_title("Split document into chunks")

            _, col2, _ = st.columns([1, 5, 1])
            with col2:
                if chunk_size := st.number_input(
                    label="Chunk size",
                    value=4000,
                    help="The max length of text segments in each chunk",
                ):
                    st.session_state[chunk_size_key] = chunk_size

                if chunk_overlap := st.number_input(
                    label="Chunk overlap",
                    value=200,
                    help="Max number of characters that should overlap between two adjacent chunks",
                ):
                    st.session_state[chunk_overlap_key] = chunk_overlap

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
                    for chunk_id, chunk_document in enumerate(chunks):
                        st.session_state[chunks_key].append(
                            Chunk(
                                id=chunk_id + 1,
                                content=chunk_document.page_content,
                                metadata=chunk_document.metadata,
                            )
                        )

            if st.session_state[chunks_key]:
                _, col2, _ = st.columns([1, 5, 1])
                with col2:
                    with st.container():
                        st.html("<br/>")
                        container_title("Chunks")
                        for chunk in st.session_state[chunks_key]:
                            with st.expander(f"CHUNK: {chunk.id}"):
                                st.text(chunk.content)

# PAGE-3: Add context to the chunks

elif st.session_state[navigation_page_key] == 3:
    if st.session_state[chunks_key]:
        with container(key="contextualised_chunks_container"):
            container_title("Add context to the chunks")

            _, col2, _ = st.columns([1, 5, 1])
            with col2:
                if not st.session_state[openai_api_key]:
                    if openai_api_key in st.secrets and st.secrets[openai_api_key]:
                        st.session_state[openai_api_key] = st.secrets[openai_api_key]

                if api_key := st.text_input(
                    label=f"{openai_api_key}",
                    value=st.session_state[openai_api_key],
                    type="password",
                ):
                    st.session_state[openai_api_key] = api_key

            _, col2, _ = st.columns([1, 0.5, 1])
            with col2:
                if st.button(
                    label="Generate",
                    type="secondary",
                    use_container_width=True,
                ):
                    if not st.session_state[openai_api_key]:
                        st.error("Please enter your OpenAI API key.", icon="🚨")
                    else:
                        parser = LlamaParse(
                            result_type=ResultType.TXT,
                            api_key=st.secrets[llama_cloud_api_key],
                            verbose=False,
                        )

                        document = " ".join(
                            [
                                doc.text
                                for doc in parser.load_data(
                                    st.session_state[document_file_path_key]
                                )
                            ]
                        )

                        def generate_context(chunk_content) -> str:
                            generate_context_prompt = ChatPromptTemplate.from_template(
                                """
                                    You are an AI assistant specializing in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document.
                                    Here is the document:
                                    <document>
                                    {document}
                                    </document>
            
                                    Here is the chunk we want to situate within the whole document:
                                    <chunk>
                                    {chunk}
                                    </chunk>
            
                                    Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
                                    1. Identify the main topic or concept discussed in the chunk.
                                    2. Mention any relevant information or comparisons from the broader document context.
                                    3. If applicable, note how this information relates to the overall theme or purpose of the document.
                                    4. Include any key figures, dates, or percentages that provide important context.
                                    5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.
            
                                    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
            
                                    Context:
                                """
                            )

                            messages = generate_context_prompt.format_messages(
                                document=document, chunk=chunk_content
                            )

                            generate_context_llm = ChatOpenAI(
                                model_name="gpt-4o",
                                openai_api_key=SecretStr(
                                    st.session_state[openai_api_key]
                                ),
                            )

                            generate_context_llm_response = generate_context_llm.invoke(
                                messages
                            )
                            return generate_context_llm_response.content

                        for chunk in st.session_state[chunks_key]:
                            chunk.context = generate_context(
                                chunk_content=chunk.content,
                            )

                            chunk.contextualized_content = (
                                f"{chunk.context}\n\n{chunk.content}"
                            )

                            st.session_state[chunks_with_context_key].append(chunk)

            if st.session_state[chunks_with_context_key]:
                _, col2, _ = st.columns([1, 5, 1])
                with col2:
                    with st.container():
                        st.html("<br/>")
                        container_title("Chunks")
                        for chunk in st.session_state[chunks_with_context_key]:
                            with st.expander(f"CHUNK: {chunk.id}"):
                                chunk_content_tab, chunk_context_tab = st.tabs(
                                    ["Chunk Content", "Chunk Context"]
                                )
                                chunk_content_tab.text(chunk.content)
                                chunk_context_tab.text(chunk.context)

# PAGE-4: Similarity Search

elif st.session_state[navigation_page_key] == 4:
    if st.session_state[chunks_with_context_key]:
        with container("similarity_search_container"):
            container_title("Similarity Search Scores")

            _, col2, _ = st.columns([1, 5, 1])
            with col2:
                st.info(
                    """
                    We will evaluate each chunk using lexical search (BM25) for precise keyword matching and vector search for semantic understanding of the query. 
                    The chunks are then ranked based on their scores, with the highest-scoring chunk assigned rank 1.
                """
                )

                st.html("<br/>")

                if user_query := st.text_input(
                    label="User Query", value=st.session_state[user_query_key]
                ):
                    st.session_state[user_query_key] = user_query

            _, col2, _ = st.columns([1, 0.2, 1])
            with col2:
                if st.button(
                    label="Search",
                    type="secondary",
                    use_container_width=True,
                ):
                    if not st.session_state[user_query_key]:
                        st.error("Please enter user query.", icon="🚨")
                    else:
                        chunks = st.session_state[chunks_with_context_key]

                        # BM25

                        tokenized_query = st.session_state[user_query_key].split()

                        chunks_content = [
                            chunk.contextualized_content for chunk in chunks
                        ]

                        tokenized_chunks_content = [
                            chunk_content.split() for chunk_content in chunks_content
                        ]

                        bm25_scores = BM25Okapi(tokenized_chunks_content).get_scores(
                            query=tokenized_query
                        )

                        bm25_ranks = len(bm25_scores) - bm25_scores.argsort().argsort()

                        for chunk, bm25_score, bm25_rank in zip(
                            chunks, bm25_scores, bm25_ranks
                        ):
                            chunk.bm25_score = bm25_score
                            chunk.bm25_rank = bm25_rank

                            st.session_state[chunks_with_bm25_scores_key].append(chunk)

                        # Vector Search

                        vector_store = FAISS.from_texts(
                            texts=[chunk.contextualized_content for chunk in chunks],
                            embedding=OpenAIEmbeddings(
                                openai_api_key=SecretStr(
                                    st.session_state["OPENAI_API_KEY"]
                                )
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

                        for chunk_with_vector_score in vector_scores:
                            for chunk in chunks:
                                if (
                                    chunk.contextualized_content
                                    == chunk_with_vector_score[0][0].page_content
                                ):
                                    chunk.vector_score = chunk_with_vector_score[0][1]
                                    chunk.vector_rank = chunk_with_vector_score[1]

                                    st.session_state[
                                        chunks_with_vector_scores_key
                                    ].append(chunk)

                        st.session_state[chunks_with_vector_scores_key] = sorted(
                            st.session_state[chunks_with_vector_scores_key],
                            key=lambda x: x.id,
                        )

            if (
                st.session_state[chunks_with_bm25_scores_key]
                and st.session_state[chunks_with_vector_scores_key]
            ):
                st.html("<br/>")

                _, col2, _ = st.columns([1, 20, 1])
                with col2:
                    bm25, vector_search = st.columns([1, 1], gap="large")

                    with bm25:
                        with st.container(border=True):
                            st.html(f"<h3 style={app_container_title_style}>BM25</h3>")

                            st.info(
                                """
                                •	BM25 is a probabilistic lexical search algorithm that relies on exact term matches between the query and documents.\n
                                •	It ranks documents based on term frequency (TF), inverse document frequency (IDF), and document length normalization.\n
                                •	Works well when the query and documents share similar words. Example: Searching for “electric car” retrieves documents with those exact words.
                            """
                            )

                            st.html("<br/>")

                            for chunk in st.session_state[chunks_with_bm25_scores_key]:
                                with st.expander(
                                    f"CHUNK: {chunk.id}, SCORE: {chunk.bm25_score}, RANK: {chunk.bm25_rank}"
                                ):
                                    chunk_content_tab, chunk_context_tab = st.tabs(
                                        ["Chunk Content", "Chunk Context"]
                                    )
                                    chunk_content_tab.text(chunk.content)
                                    chunk_context_tab.text(chunk.context)

                    with vector_search:
                        with st.container(border=True):
                            st.html(
                                f"<h3 style={app_container_title_style}>Vector Search</h3>"
                            )

                            st.info(
                                """
                                •	A semantic search method that uses embeddings (dense vectors) to represent the meaning of queries and documents.\n
                                •	Matches based on similarity in vector space (e.g., cosine similarity or dot product), capturing semantic relationships even if words don’t overlap.\n
                                •	Example: Searching for “electric vehicle” retrieves documents mentioning “EV” or “battery-powered car.”
                            """
                            )

                            st.html("<br/>")

                            for chunk in st.session_state[
                                chunks_with_vector_scores_key
                            ]:
                                with st.expander(
                                    f"CHUNK: {chunk.id}, SCORE: {chunk.vector_score}, RANK: {chunk.vector_rank}"
                                ):
                                    chunk_content_tab, chunk_context_tab = st.tabs(
                                        ["Chunk Content", "Chunk Context"]
                                    )
                                    chunk_content_tab.text(chunk.content)
                                    chunk_context_tab.text(chunk.context)

# PAGE-5: Re-Ranking

elif st.session_state[navigation_page_key] == 5:
    if (
        st.session_state[chunks_with_bm25_scores_key]
        and st.session_state[chunks_with_vector_scores_key]
    ):
        with container("reranking_container"):
            container_title("Reciprocal Rank Fusion (RRF)")

            _, col2, _ = st.columns([1, 5, 1])
            with col2:
                st.info(
                    """
                    •	Reciprocal Rank Fusion (RRF) is a simple and effective method used in information retrieval to combine results from multiple ranked lists, such as those generated by different retrieval methods (e.g., BM25 and vector search). 
                        It assigns a combined score to each item based on its ranks in the individual lists.
                    \n
                    •	The formula for RRF is as follows:\n
                        RRF = 1 / (k + rank_bm25) + 1 / (k + rank_vector)\n
                        where k is a constant (60 in this case) and rank_bm25 and rank_vector are the ranks of the item in the BM25 and vector search lists, respectively.
                """
                )

                st.html("<br/>")

                def rrf_score(keyword_rank: int, semantic_rank: int) -> float:
                    k = 60
                    return 1 / (k + keyword_rank) + 1 / (k + semantic_rank)

                chunk_with_rrf_scores = [
                    Chunk(
                        id=chunk_with_bm25_rank.id,
                        content=chunk_with_bm25_rank.content,
                        metadata=chunk_with_bm25_rank.metadata,
                        context=chunk_with_bm25_rank.context,
                        contextualized_content=chunk_with_bm25_rank.contextualized_content,
                        bm25_score=chunk_with_bm25_rank.bm25_score,
                        bm25_rank=chunk_with_bm25_rank.bm25_rank,
                        vector_score=chunk_with_vector_rank.vector_score,
                        vector_rank=chunk_with_vector_rank.vector_rank,
                        rrf_score=rrf_score(
                            chunk_with_bm25_rank.bm25_score,
                            chunk_with_vector_rank.vector_score,
                        ),
                    )
                    for chunk_with_bm25_rank, chunk_with_vector_rank in zip(
                        st.session_state[chunks_with_bm25_scores_key],
                        st.session_state[chunks_with_vector_scores_key],
                    )
                ]

                rrf_ranks = (
                    len(chunk_with_rrf_scores)
                    - np.array(list(map(lambda x: x.rrf_score, chunk_with_rrf_scores)))
                    .argsort()
                    .argsort()
                )

                for chunk, rrf_rank in zip(chunk_with_rrf_scores, rrf_ranks):
                    chunk.rrf_rank = rrf_rank
                    st.session_state[chunks_with_rrf_scores_key].append(chunk)

                st.session_state[chunks_with_rrf_scores_key] = sorted(
                    st.session_state[chunks_with_rrf_scores_key],
                    key=lambda x: x.id,
                )

                for chunk in st.session_state[chunks_with_rrf_scores_key]:
                    with st.expander(
                        f"CHUNK: {chunk.id}, SCORE: {chunk.rrf_score}, RANK: {chunk.rrf_rank}"
                    ):
                        chunk_content_tab, chunk_context_tab = st.tabs(
                            ["Chunk Content", "Chunk Context"]
                        )
                        chunk_content_tab.text(chunk.content)
                        chunk_context_tab.text(chunk.context)

# PAGE-6: Generate Response

elif st.session_state[navigation_page_key] == 6:
    if st.session_state[chunks_with_rrf_scores_key]:
        container_title("Generate Response")

        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            with container("llm_input_container"):
                container_title("LLM Input")

                _, llm_input, _ = st.columns([1, 10, 1])
                with llm_input:
                    st.info(
                        """
                        •	Top 4 chunks are selected based on RRF scores.\n
                        •	These chunks are then passed to the LLM along with the user query to generate the response.
                    """
                    )

                    st.html("<br/>")

                    with st.container(border=True):
                        container_title("User Query")
                        st.write(st.session_state[user_query_key])

                    st.html("<br/>")

                    with st.container(border=True):
                        container_title("LLM Context")

                        top_4_chunks = sorted(
                            st.session_state[chunks_with_rrf_scores_key],
                            key=lambda x: x.rrf_rank,
                        )[0:4]

                        for chunk in top_4_chunks:
                            with st.expander(
                                f"CHUNK: {chunk.id}, SCORE: {chunk.rrf_score}, RANK: {chunk.rrf_rank}"
                            ):
                                chunk_content_tab, chunk_context_tab = st.tabs(
                                    ["Chunk Content", "Chunk Context"]
                                )
                                chunk_content_tab.text(chunk.content)
                                chunk_context_tab.text(chunk.context)

        with col2:
            with container("llm_output_container"):
                container_title("LLM Output")

                response = None

                col1, col2, col3 = st.columns([1, 0.8, 1])
                with col2:
                    if st.button(
                        label="Generate Response",
                        type="secondary",
                        use_container_width=True,
                    ):
                        prompt = Prompt.from_template(
                            """
                            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
                            Question: {question} 
    
                            Context: {context} 
    
                            Answer:
                            """
                        )

                        llm = ChatOpenAI(
                            model_name="gpt-4o",
                            openai_api_key=SecretStr(st.secrets["OPENAI_API_KEY"]),
                        )

                        rag_chain = prompt | llm

                        llm_context = "\n\n".join(
                            chunk.contextualized_content for chunk in top_4_chunks
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

                st.html("<br/>")

                _, llm_output, _ = st.columns([1, 10, 1])
                with llm_output:
                    with st.container(border=True):
                        container_title("LLM Response")

                        if response:
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
    elif st.session_state[navigation_page_key] == 4:
        if (
            not st.session_state[chunks_with_bm25_scores_key]
            or not st.session_state[chunks_with_vector_scores_key]
        ):
            next_button_disabled = True
    elif st.session_state[navigation_page_key] == 5:
        if not st.session_state[chunks_with_rrf_scores_key]:
            next_button_disabled = True
    elif st.session_state[navigation_page_key] == 6:
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
