import tempfile
import time

import streamlit as st
from langchain.prompts import Prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import SecretStr

file_uploader_key = "gr_file_uploader"

uploaded_file_key = "gr_uploaded_file"
if uploaded_file_key not in st.session_state:
    st.session_state[uploaded_file_key] = None

loaded_pdf_pages_key = "gr_pdf_pages"
if loaded_pdf_pages_key not in st.session_state:
    st.session_state[loaded_pdf_pages_key] = None

graph_documents_key = "gr_graph_documents"
if graph_documents_key not in st.session_state:
    st.session_state[graph_documents_key] = None

navigation_page_key = "gr_navigation_page"
if navigation_page_key not in st.session_state:
    st.session_state[navigation_page_key] = 1

openai_api_key = "OPENAI_API_KEY"
if openai_api_key not in st.session_state:
    st.session_state[openai_api_key] = ""

neo4j_uri_key = "NEO4J_URI"
if neo4j_uri_key not in st.session_state:
    st.session_state[neo4j_uri_key] = ""

neo4j_username_key = "NEO4J_USERNAME"
if neo4j_username_key not in st.session_state:
    st.session_state[neo4j_username_key] = ""

neo4j_password_key = "NEO4J_PASSWORD"
if neo4j_password_key not in st.session_state:
    st.session_state[neo4j_password_key] = ""

neo4j_database_key = "NEO4J_DATABASE"
if neo4j_database_key not in st.session_state:
    st.session_state[neo4j_database_key] = "neo4j"

user_query_key = "gr_user_query"
if user_query_key not in st.session_state:
    st.session_state[user_query_key] = ""

st.set_page_config(
    page_title="Graph RAG",
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
        
        .custom-container {
            background-color: #f0f8ff; /* Light blue */
        }
    </style>
    """
)

st.html(
    f"<h1 style='font-family:Poppins; color:#C4DAD2; text-align: center'>Graph RAG</h1><br/>"
)

with st.container(border=True):
    # PAGE-1: Upload PDF

    if st.session_state[navigation_page_key] == 1:
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

    # PAGE-2: Convert PDF into graph documents

    elif st.session_state[navigation_page_key] == 2:
        if st.session_state[loaded_pdf_pages_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Convert PDF into Graph documents</h3>"
            )

            if not st.session_state[openai_api_key]:
                if openai_api_key in st.secrets and st.secrets[openai_api_key]:
                    st.session_state[openai_api_key] = st.secrets[openai_api_key]
                else:
                    st.error("Please enter your OpenAI API key.", icon="🚨")

            if api_key := st.text_input(
                label=f"{openai_api_key}",
                value=st.session_state[openai_api_key],
                type="password",
            ):
                st.session_state[openai_api_key] = api_key

            if st.session_state[openai_api_key]:
                st.html("<br/>")
                _, col2, _ = st.columns([1, 1, 1])
                with col2:
                    if st.button(
                        label="Convert",
                        type="secondary",
                        use_container_width=True,
                    ):
                        llm = ChatOpenAI(
                            model_name="gpt-4o",
                            openai_api_key=st.session_state[openai_api_key],
                            temperature=0,
                        )

                        graph_documents = LLMGraphTransformer(
                            llm=llm
                        ).convert_to_graph_documents(
                            documents=st.session_state[loaded_pdf_pages_key]
                        )

                        st.session_state[graph_documents_key] = graph_documents

        if st.session_state[graph_documents_key]:
            st.html("<br/>")

            with st.container(border=True):
                st.html(
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>Graph Documents</p>"
                )

                for idx, graph_document in enumerate(
                    st.session_state[graph_documents_key]
                ):
                    with st.expander(
                        f"GRAPH DOCUMENT: {idx + 1}, NODES: {len(graph_document.nodes)}, RELATIONSHIPS: {len(graph_document.relationships)}"
                    ):
                        graph_document_nodes, graph_document_relationships = st.tabs(
                            ["NODES", "RELATIONSHIPS"]
                        )
                        with graph_document_nodes:
                            nodes = []
                            for node in graph_document.nodes:
                                flattened_node = {
                                    "ID": node.id,
                                    "TYPE": node.type,
                                }
                                nodes.append(flattened_node)
                            st.table(nodes)
                        with graph_document_relationships:
                            relationships = []
                            for relationship in graph_document.relationships:
                                flattened_relationship = {
                                    "SOURCE": relationship.source.id,
                                    "TARGET": relationship.target.id,
                                    "RELATIONSHIP": relationship.type,
                                }
                                relationships.append(flattened_relationship)
                            st.table(relationships)

    # PAGE-3: Store graph documents to Graph DB

    elif st.session_state[navigation_page_key] == 3:
        if st.session_state[graph_documents_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Load graph documents to Graph DB</h3>"
            )

            # NEO4J_URI

            if not st.session_state[neo4j_uri_key]:
                if neo4j_uri_key in st.secrets and st.secrets[neo4j_uri_key]:
                    st.session_state[neo4j_uri_key] = st.secrets[neo4j_uri_key]
                else:
                    st.error(f"Please enter {neo4j_uri_key}", icon="🚨")

            if neo4j_uri := st.text_input(
                label=f"{neo4j_uri_key}", value=st.session_state[neo4j_uri_key]
            ):
                st.session_state[neo4j_uri_key] = neo4j_uri

            # NEO4J_USERNAME

            if not st.session_state[neo4j_username_key]:
                if neo4j_username_key in st.secrets and st.secrets[neo4j_username_key]:
                    st.session_state[neo4j_username_key] = st.secrets[
                        neo4j_username_key
                    ]
                else:
                    st.error(f"Please enter {neo4j_username_key}", icon="🚨")

            if neo4j_username := st.text_input(
                label=f"{neo4j_username_key}",
                value=st.session_state[neo4j_username_key],
            ):
                st.session_state[neo4j_username_key] = neo4j_username

            # NEO4J_PASSWORD

            if not st.session_state[neo4j_password_key]:
                if neo4j_password_key in st.secrets and st.secrets[neo4j_password_key]:
                    st.session_state[neo4j_password_key] = st.secrets[
                        neo4j_password_key
                    ]
                else:
                    st.error(f"Please enter {neo4j_password_key}", icon="🚨")

            if neo4j_password := st.text_input(
                label=f"{neo4j_password_key}",
                value=st.session_state[neo4j_password_key],
                type="password",
            ):
                st.session_state[neo4j_password_key] = neo4j_password

            # NEO4J_DATABASE

            if not st.session_state[neo4j_database_key]:
                if neo4j_database_key in st.secrets and st.secrets[neo4j_database_key]:
                    st.session_state[neo4j_database_key] = st.secrets[
                        neo4j_database_key
                    ]
                else:
                    st.error(f"Please enter {neo4j_database_key}", icon="🚨")

            if neo4j_database := st.text_input(
                label=f"{neo4j_database_key}",
                value=st.session_state[neo4j_database_key],
            ):
                st.session_state[neo4j_database_key] = neo4j_database

            st.html("<br/>")

            _, col2, _ = st.columns([1, 1, 1])

            with col2:
                if st.button(
                    label="Load",
                    type="secondary",
                    use_container_width=True,
                ):
                    graph = Neo4jGraph(
                        url=neo4j_uri,
                        username=neo4j_username,
                        password=neo4j_password,
                        database=neo4j_database,
                    )

                    graph.add_graph_documents(
                        graph_documents=st.session_state[graph_documents_key]
                    )

    # PAGE-4: Similarity Search

    elif st.session_state[navigation_page_key] == 4:
        if st.session_state[chunks_with_embeddings_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Similarity Search</h3>"
            )

            if user_query := st.text_input(
                label="User Query", value=st.session_state[user_query_key]
            ):
                st.session_state[user_query_key] = user_query

            st.html("<br/>")
            _, col2, _ = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    label="Search",
                    type="secondary",
                    use_container_width=True,
                ):
                    if not st.session_state[user_query_key]:
                        st.error("Please enter user query.", icon="🚨")
                    else:
                        st.session_state[similar_chunks_key] = []

                        vector_store = FAISS.from_documents(
                            documents=[
                                chunk_info["chunk"]
                                for chunk_info in st.session_state[
                                    chunks_with_embeddings_key
                                ]
                            ],
                            embedding=OpenAIEmbeddings(
                                openai_api_key=SecretStr(
                                    st.session_state["OPENAI_API_KEY"]
                                )
                            ),
                        )

                        similar_chunks = vector_store.similarity_search_with_score(
                            user_query
                        )

                        similar_chunks_sorted_by_score = sorted(
                            similar_chunks, key=lambda x: x[1], reverse=True
                        )

                        st.session_state[similar_chunks_key] = (
                            similar_chunks_sorted_by_score
                        )

        if st.session_state[similar_chunks_key]:
            st.html("<br/>")

            with st.container(border=True):
                st.html(
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>User Query Embedding</p>"
                )

                embeddings = OpenAIEmbeddings(
                    openai_api_key=SecretStr(st.session_state[openai_api_key]),
                )

                user_query_embedding = embeddings.embed_query(
                    st.session_state[user_query_key]
                )

                with st.expander(label="User Query Embedding"):
                    st.text(user_query_embedding)

                st.html("<br/>")
                st.html(
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>Similar Chunks</p>"
                )

                for similar_chunk, score in st.session_state[similar_chunks_key]:
                    for chunk_with_embedding in st.session_state[
                        chunks_with_embeddings_key
                    ]:
                        if chunk_with_embedding["chunk"] == similar_chunk:
                            with st.expander(
                                f"Chunk: {chunk_with_embedding["chunk_id"]}, Score: {score}"
                            ):
                                chunk_content, chunk_embedding = st.tabs(
                                    ["Chunk Content", "Chunk Embedding"]
                                )
                                chunk_content.text(
                                    chunk_with_embedding["chunk"].page_content
                                )
                                chunk_embedding.text(
                                    chunk_with_embedding["chunk_embedding"]
                                )

    # PAGE-5: Generate Response

    elif st.session_state[navigation_page_key] == 5:
        if st.session_state[similar_chunks_key]:
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

                        for similar_chunk, score in st.session_state[
                            similar_chunks_key
                        ]:
                            for chunk_with_embedding in st.session_state[
                                chunks_with_embeddings_key
                            ]:
                                if chunk_with_embedding["chunk"] == similar_chunk:
                                    with st.expander(
                                        f"Chunk: {chunk_with_embedding["chunk_id"]}, Score: {score}"
                                    ):
                                        st.text(
                                            chunk_with_embedding["chunk"].page_content
                                        )

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
                                openai_api_key=st.session_state[openai_api_key],
                                temperature=0,
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
                                similar_chunk.page_content
                                for (similar_chunk, _) in st.session_state[
                                    similar_chunks_key
                                ]
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

container = st.container()
previous_col, _, next_col = container.columns([1, 1, 1])

previous_button_disabled = False
next_button_disabled = False

if st.session_state[navigation_page_key] == 1:
    previous_button_disabled = True
    if not st.session_state[uploaded_file_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 2:
    if not st.session_state[graph_documents_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 3:
    if not st.session_state[graph_documents_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 4:
    if not st.session_state[similar_chunks_key]:
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
