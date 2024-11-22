import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from agents.simple_rag_agent import SimpleRAGAgent

required_keys = {"OPENAI_API_KEY": "password"}

uploaded_file_key = "sr_uploaded_file"
if uploaded_file_key not in st.session_state:
    st.session_state[uploaded_file_key] = None

agent = SimpleRAGAgent


def required_keys_missing():
    if_missing = False
    for key in required_keys.keys():
        if key not in st.session_state or not st.session_state[key]:
            st.error(f"Please enter {key}", icon="🚨")
            if_missing = True
    return if_missing


def is_first_human_message():
    for page_message in st.session_state.page_messages[agent.agent_name]:
        if page_message.get("role") == "human":
            return False
    return True


def add_chat_message(role: str, content: str):
    st.session_state.page_messages[agent.agent_name].append(
        {"role": role, "content": content}
    )
    with st.chat_message(role):
        st.markdown(f"<p class='fontStyle'>{content}</p>", unsafe_allow_html=True)


def store_document_in_vector_store(document):
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.write(document.read())

    loaded_pdf = PyPDFLoader(file_path=temp_file.name).load()

    documents = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(loaded_pdf)

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(
            openai_api_key=SecretStr(st.session_state["OPENAI_API_KEY"])
        ),
    )

    return vector_store


st.set_page_config(page_title=agent.agent_name, page_icon="🤖", layout="wide")

st.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        .stApp {
            font-family: 'Poppins';
            background-color: #16423C;
        }
        
        .fontStyle {
            font-family: 'Poppins';
        }
        
        [title="Show password text"] {
            display: none;
        }
    </style>
    """
)

st.html(f"<h2 class='fontStyle' style='color:#C4DAD2';>{agent.agent_name}</h2><br/>")

with st.sidebar:
    for key_name, key_type in required_keys.items():
        if key_name not in st.session_state:
            st.session_state[key_name] = None

        if not st.session_state[key_name]:
            if key_name in st.secrets and st.secrets[key_name]:
                st.session_state[key_name] = st.secrets[key_name]

        if api_key := st.text_input(
            label=f"{key_name}", value=st.session_state[key_name], type=key_type
        ):
            st.session_state[key_name] = api_key

if required_keys and not required_keys_missing():
    with st.sidebar:
        st.divider()

        # FILE UPLOADER

        st.html(
            "<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>Upload PDF File</h3>"
        )

        if uploaded_file := st.file_uploader(
            label="Upload PDF File", label_visibility="hidden", type=["pdf"]
        ):
            if (
                not st.session_state[uploaded_file_key]
                or st.session_state[uploaded_file_key] != uploaded_file
            ):
                st.session_state[uploaded_file_key] = uploaded_file
                st.session_state["sr_vector_store"] = store_document_in_vector_store(
                    uploaded_file
                )

        if not uploaded_file and st.session_state[uploaded_file_key]:
            st.success(
                f"Uploaded: {st.session_state[uploaded_file_key].name}", icon="✅"
            )

        st.divider()

        # LANGGRAPH WORKFLOW

        st.markdown(
            "<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>LangGraph Workflow Visualization</h3>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
                [data-testid="stImage"] {
                    border-radius: 10px;
                    overflow: hidden;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        agent_graph = agent.get_graph()

        st.image(
            agent_graph.get_graph(xray=1).draw_mermaid_png(),
            use_container_width=True,
        )

    if "page_messages" not in st.session_state:
        st.session_state.page_messages = {}

    if agent.agent_name not in st.session_state.page_messages:
        st.session_state.page_messages[agent.agent_name] = []

    for message in st.session_state.page_messages[agent.agent_name]:
        with st.chat_message(message["role"]):
            st.markdown(
                f"<p class='fontStyle'>{message["content"]}</p>",
                unsafe_allow_html=True,
            )

    if human_message := st.chat_input():
        if not st.session_state[uploaded_file_key]:
            st.error("Please upload a file before sending a message.", icon="🚨")
        else:
            add_chat_message(role="human", content=human_message)

            if is_first_human_message():
                agent_input = {
                    "messages": [
                        SystemMessage(content=agent.system_prompt),
                        HumanMessage(content=human_message),
                    ]
                }
            else:
                agent_input = {
                    "messages": [
                        HumanMessage(content=human_message),
                    ]
                }

            config = {"configurable": {"thread_id": "1"}}

            for event in agent_graph.stream(
                input=agent_input,
                config=config,
                stream_mode="updates",
            ):
                for k, v in event.items():
                    if k in agent.nodes_to_display:
                        if "messages" in v:
                            m = v["messages"][-1]
                            if (
                                m.type == "ai" and not m.tool_calls
                            ) or m.type == "human":
                                add_chat_message(role=m.type, content=m.content)
