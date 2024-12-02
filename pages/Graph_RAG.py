import tempfile
import time

import streamlit as st
from graphviz import Digraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import Prompt

file_uploader_key = "gr_file_uploader"

uploaded_file_key = "gr_uploaded_file"
if uploaded_file_key not in st.session_state:
    st.session_state[uploaded_file_key] = None

loaded_pdf_pages_key = "gr_pdf_pages"
if loaded_pdf_pages_key not in st.session_state:
    st.session_state[loaded_pdf_pages_key] = None

chunk_size_key = "gr_chunk_size"
if chunk_size_key not in st.session_state:
    st.session_state[chunk_size_key] = None

chunk_overlap_key = "gr_chunk_overlap"
if chunk_overlap_key not in st.session_state:
    st.session_state[chunk_overlap_key] = None

chunks_key = "gr_chunks"
if chunks_key not in st.session_state:
    st.session_state[chunks_key] = []

chunks_with_graph_key = "gr_chunks_with_graph"
if chunks_with_graph_key not in st.session_state:
    st.session_state[chunks_with_graph_key] = []

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

vector_store_key = "gr_vector_store"
if vector_store_key not in st.session_state:
    st.session_state[vector_store_key] = None

user_query_key = "gr_user_query"
if user_query_key not in st.session_state:
    st.session_state[user_query_key] = ""

similar_chunks_key = "gr_similar_chunks"
if similar_chunks_key not in st.session_state:
    st.session_state[similar_chunks_key] = []

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
    # PAGE-1: Upload document

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

    # PAGE-2: Split document into chunks

    elif st.session_state[navigation_page_key] == 2:
        if st.session_state[uploaded_file_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Split document into chunks</h3>"
            )

            if chunk_size := st.number_input(label="Chunk size", value=4000):
                st.session_state[chunk_size_key] = chunk_size

            if chunk_overlap := st.number_input(label="Chunk overlap", value=200):
                st.session_state[chunk_overlap_key] = chunk_overlap

            st.html("<br/>")

            _, col2, _ = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    label="Create chunks", type="secondary", use_container_width=True
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

    # PAGE-3: Convert chunks into graphs

    elif st.session_state[navigation_page_key] == 3:
        if st.session_state[chunks_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Convert chunks into graphs</h3>"
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

                        st.session_state[chunks_with_graph_key] = []

                        for chunk_info in st.session_state[chunks_key]:
                            chunk_graph = LLMGraphTransformer(
                                llm=llm
                            ).convert_to_graph_documents(
                                documents=[chunk_info["chunk"]]
                            )

                            chunk_info["chunk_graph"] = chunk_graph[0]

                            st.session_state[chunks_with_graph_key].append(chunk_info)

        if st.session_state[chunks_with_graph_key]:
            st.html("<br/>")

            with st.container(border=True):
                st.html(
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>Chunks with graphs</p>"
                )

                for chunk_info in st.session_state[chunks_with_graph_key]:
                    with st.expander(
                        f"CHUNK: {chunk_info["chunk_id"]}, NODES: {len(chunk_info["chunk_graph"].nodes)}, RELATIONSHIPS: {len(chunk_info["chunk_graph"].relationships)}"
                    ):
                        (
                            chunk_content,
                            chunk_graph_nodes,
                            chunk_graph_relationships,
                            chunk_graph,
                        ) = st.tabs(["CONTENT", "NODES", "RELATIONSHIPS", "GRAPH"])
                        with chunk_content:
                            st.text(chunk_info["chunk"].page_content)
                        with chunk_graph_nodes:
                            nodes = []
                            for node in chunk_info["chunk_graph"].nodes:
                                flattened_node = {
                                    "ID": node.id,
                                    "TYPE": node.type,
                                }
                                nodes.append(flattened_node)
                            st.table(nodes)
                        with chunk_graph_relationships:
                            relationships = []
                            for relationship in chunk_info["chunk_graph"].relationships:
                                flattened_relationship = {
                                    "SOURCE": relationship.source.id,
                                    "RELATIONSHIP": relationship.type,
                                    "TARGET": relationship.target.id,
                                }
                                relationships.append(flattened_relationship)
                            st.table(relationships)
                        with chunk_graph:
                            graph = Digraph()
                            for node in chunk_info["chunk_graph"].nodes:
                                graph.node(node.id, node.id)
                            for relationship in chunk_info["chunk_graph"].relationships:
                                graph.edge(
                                    relationship.source.id,
                                    relationship.target.id,
                                    relationship.type,
                                )
                            st.graphviz_chart(graph)

    # PAGE-4: Load chunks to Graph DB (Neo4j)

    elif st.session_state[navigation_page_key] == 4:
        if st.session_state[chunks_with_graph_key]:
            st.html(
                "<h3 style='color:#E9EFEC; font-family:Poppins; text-align: center'>Load chunks to Graph DB (Neo4j)</h3>"
            )

            # NEO4J_URI

            if not st.session_state[neo4j_uri_key]:
                if neo4j_uri_key in st.secrets and st.secrets[neo4j_uri_key]:
                    st.session_state[neo4j_uri_key] = st.secrets[neo4j_uri_key]
                else:
                    st.error(f"Please enter {neo4j_uri_key}", icon="🚨")

            if neo4j_uri := st.text_input(
                label=f"{neo4j_uri_key}",
                value=st.session_state[neo4j_uri_key],
                disabled=True,
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
                disabled=True,
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
                disabled=True,
            ):
                st.session_state[neo4j_password_key] = neo4j_password

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
                    )

                    for chunk_info in st.session_state[chunks_with_graph_key]:
                        graph.add_graph_documents(
                            graph_documents=[chunk_info["chunk_graph"]],
                            include_source=True,
                            baseEntityLabel=True,
                        )

                    vector_store = Neo4jVector.from_existing_graph(
                        graph=graph,
                        embedding=OpenAIEmbeddings(
                            openai_api_key=st.session_state[openai_api_key]
                        ),
                        node_label="Document",
                        text_node_properties=["text"],
                        embedding_node_property="embedding",
                    )

                    st.session_state[vector_store_key] = vector_store

            if st.session_state[vector_store_key]:
                st.html("<br/>")
                st.success("Chunks are successfully loaded to Neo4j", icon="✅")

    # PAGE-5: Similarity Search

    elif st.session_state[navigation_page_key] == 5:
        if st.session_state[vector_store_key]:
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
                        vector_store = st.session_state[vector_store_key]

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
                    "<p style='color:#E9EFEC; font-family:Poppins; text-align: center'>Similar Chunks</p>"
                )

                for similar_chunk, score in st.session_state[similar_chunks_key]:
                    for chunk_with_graph in st.session_state[chunks_with_graph_key]:
                        if (
                            similar_chunk.page_content.split("\ntext: ")[1]
                            in chunk_with_graph["chunk"].page_content
                        ):
                            with st.expander(
                                f"CHUNK: {chunk_with_graph["chunk_id"]}, NODES: {len(chunk_with_graph["chunk_graph"].nodes)}, RELATIONSHIPS: {len(chunk_with_graph["chunk_graph"].relationships)}, SCORE: {score}"
                            ):
                                (
                                    chunk_content,
                                    chunk_graph_nodes,
                                    chunk_graph_relationships,
                                    chunk_graph,
                                ) = st.tabs(
                                    ["CONTENT", "NODES", "RELATIONSHIPS", "GRAPH"]
                                )
                                with chunk_content:
                                    st.text(chunk_with_graph["chunk"].page_content)
                                with chunk_graph_nodes:
                                    nodes = []
                                    for node in chunk_with_graph["chunk_graph"].nodes:
                                        flattened_node = {
                                            "ID": node.id,
                                            "TYPE": node.type,
                                        }
                                        nodes.append(flattened_node)
                                    st.table(nodes)
                                with chunk_graph_relationships:
                                    relationships = []
                                    for relationship in chunk_with_graph[
                                        "chunk_graph"
                                    ].relationships:
                                        flattened_relationship = {
                                            "SOURCE": relationship.source.id,
                                            "RELATIONSHIP": relationship.type,
                                            "TARGET": relationship.target.id,
                                        }
                                        relationships.append(flattened_relationship)
                                    st.table(relationships)
                                with chunk_graph:
                                    graph = Digraph()
                                    for node in chunk_with_graph["chunk_graph"].nodes:
                                        graph.node(node.id, node.id)
                                    for relationship in chunk_with_graph[
                                        "chunk_graph"
                                    ].relationships:
                                        graph.edge(
                                            relationship.source.id,
                                            relationship.target.id,
                                            relationship.type,
                                        )
                                    st.graphviz_chart(graph)

    # PAGE-6: Generate Response

    elif st.session_state[navigation_page_key] == 6:
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
                            for chunk_with_graph in st.session_state[
                                chunks_with_graph_key
                            ]:
                                if (
                                    similar_chunk.page_content.split("\ntext: ")[1]
                                    in chunk_with_graph["chunk"].page_content
                                ):
                                    with st.expander(
                                        f"CHUNK: {chunk_with_graph["chunk_id"]}, NODES: {len(chunk_with_graph["chunk_graph"].nodes)}, RELATIONSHIPS: {len(chunk_with_graph["chunk_graph"].relationships)}, SCORE: {score}"
                                    ):
                                        (
                                            chunk_content,
                                            chunk_graph_nodes,
                                            chunk_graph_relationships,
                                            chunk_graph,
                                        ) = st.tabs(
                                            [
                                                "CONTENT",
                                                "NODES",
                                                "RELATIONSHIPS",
                                                "GRAPH",
                                            ]
                                        )
                                        with chunk_content:
                                            st.text(
                                                chunk_with_graph["chunk"].page_content
                                            )
                                        with chunk_graph_nodes:
                                            nodes = []
                                            for node in chunk_with_graph[
                                                "chunk_graph"
                                            ].nodes:
                                                flattened_node = {
                                                    "ID": node.id,
                                                    "TYPE": node.type,
                                                }
                                                nodes.append(flattened_node)
                                            st.table(nodes)
                                        with chunk_graph_relationships:
                                            relationships = []
                                            for relationship in chunk_with_graph[
                                                "chunk_graph"
                                            ].relationships:
                                                flattened_relationship = {
                                                    "SOURCE": relationship.source.id,
                                                    "RELATIONSHIP": relationship.type,
                                                    "TARGET": relationship.target.id,
                                                }
                                                relationships.append(
                                                    flattened_relationship
                                                )
                                            st.table(relationships)
                                        with chunk_graph:
                                            graph = Digraph()
                                            for node in chunk_with_graph[
                                                "chunk_graph"
                                            ].nodes:
                                                graph.node(node.id, node.id)
                                            for relationship in chunk_with_graph[
                                                "chunk_graph"
                                            ].relationships:
                                                graph.edge(
                                                    relationship.source.id,
                                                    relationship.target.id,
                                                    relationship.type,
                                                )
                                            st.graphviz_chart(graph)

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
                                doc.page_content
                                for doc in st.session_state[vector_store_key]
                                .as_retriever()
                                .invoke(st.session_state[user_query_key])
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
    if not st.session_state[chunks_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 3:
    if not st.session_state[chunks_with_graph_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 4:
    if not st.session_state[vector_store_key]:
        next_button_disabled = True
elif st.session_state[navigation_page_key] == 5:
    if not st.session_state[similar_chunks_key]:
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
