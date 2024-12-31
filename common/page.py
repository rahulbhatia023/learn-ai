import tempfile

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from streamlit.commands.page_config import Layout, PageIcon

from common.agent import BaseAgent
from common.chat import add_chat_message, display_message


def get_api_key(keys):
    for key_name, key_type in keys.items():
        if key_name not in st.session_state:
            st.session_state[key_name] = None

        if not st.session_state[key_name]:
            if key_name in st.secrets and st.secrets[key_name]:
                st.session_state[key_name] = st.secrets[key_name]

        if api_key := st.text_input(
            label=f"{key_name}", value=st.session_state[key_name], type=key_type
        ):
            st.session_state[key_name] = api_key


def keys_missing(keys):
    if_missing = False
    for key in keys.keys():
        if key not in st.session_state or not st.session_state[key]:
            st.error(f"Please enter {key}", icon="🚨")
            if_missing = True
    return if_missing


class BasePage:
    agent: BaseAgent = None
    required_keys: dict[str, str] = {}
    page_icon: PageIcon = "🤖"
    layout: Layout = "wide"

    show_file_uploader: bool = False
    file_upload_label: str = "Upload a file"
    file_upload_type: list[str] = ["csv"]

    @classmethod
    def on_file_upload(cls, uploaded_file):
        pass

    @classmethod
    def stream_events(cls, agent_graph, human_message):
        config = {"configurable": {"thread_id": "1"}}

        def is_first_human_message():
            for message in st.session_state.page_messages[cls.agent.name]:
                if message.get("role") == "human":
                    return False
            return True

        if human_message:
            if is_first_human_message():
                agent_input = {
                    "messages": [
                        SystemMessage(content=cls.agent.system_prompt),
                        HumanMessage(content=human_message),
                    ]
                }
            elif not cls.agent.interrupt_before:
                agent_input = {
                    "messages": [
                        HumanMessage(content=human_message),
                    ]
                }
            else:
                agent_input = None
                agent_graph.update_state(
                    config=config,
                    values={"messages": [HumanMessage(content=human_message)]}
                    | cls.agent.update_graph_state(human_message),
                    as_node=cls.agent.update_as_node,
                )

            add_chat_message(
                agent_name=cls.agent.name, role="human", content=human_message
            )

            with st.spinner(text="Thinking..."):
                ai_messages = []
                for event in agent_graph.stream(
                    input=agent_input,
                    config=config,
                    stream_mode="updates",
                ):
                    for k, v in event.items():
                        if cls.agent.nodes_to_display:
                            if k in cls.agent.nodes_to_display:
                                ai_messages.append(v)

                        else:
                            ai_messages.append(v)

            for message in ai_messages:
                display_message(agent_name=cls.agent.name, v=message)

    @classmethod
    def pre_render(cls):
        pass

    @classmethod
    def render(cls):
        st.set_page_config(
            page_title=cls.agent.name, page_icon=cls.page_icon, layout=cls.layout
        )

        st.markdown(
            """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins');

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
            """,
            unsafe_allow_html=True,
        )

        st.html(
            f"<h2 class='fontStyle' style='color:#C4DAD2';>{cls.agent.name}</h2><br/>"
        )

        with st.sidebar:
            get_api_key(cls.required_keys)

        if cls.required_keys and not keys_missing(cls.required_keys):
            if "agent_graph" not in st.session_state:
                st.session_state.agent_graph = {}
            if cls.agent.name not in st.session_state.agent_graph:
                st.session_state.agent_graph[cls.agent.name] = None
            if not st.session_state.agent_graph[cls.agent.name]:
                st.session_state.agent_graph[cls.agent.name] = cls.agent.get_graph()

            agent_graph = st.session_state.agent_graph[cls.agent.name]

            with st.sidebar:
                st.divider()

                st.html(
                    "<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>LangGraph Workflow Visualization</h3>"
                )

                st.html(
                    """
                    <style>
                        [data-testid="stImage"] {
                            border-radius: 10px;
                            overflow: hidden;
                        }
                    </style>
                    """
                )

                st.image(
                    agent_graph.get_graph(xray=1).draw_mermaid_png(),
                    use_container_width=True,
                )

                if cls.show_file_uploader:
                    if "uploaded_file" not in st.session_state:
                        st.session_state.uploaded_file = {}
                    if cls.agent.name not in st.session_state.uploaded_file:
                        st.session_state.uploaded_file[cls.agent.name] = None

                    st.html(
                        f"<br/><br/><h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>{cls.file_upload_label}</h3>"
                    )

                    if uploaded_file := st.file_uploader(
                        label=cls.file_upload_label,
                        type=cls.file_upload_type,
                        label_visibility="hidden",
                    ):
                        if not st.session_state["uploaded_file"][cls.agent.name]:
                            st.info("Uploading file, please wait...")
                            with tempfile.NamedTemporaryFile(delete=False) as file:
                                file.write(uploaded_file.read())
                                file.flush()
                                st.session_state["uploaded_file"][
                                    cls.agent.name
                                ] = file.name

                            cls.on_file_upload(
                                uploaded_file=st.session_state["uploaded_file"][
                                    cls.agent.name
                                ]
                            )
                            st.info("File uploaded successfully")

                        agent_graph = cls.agent.get_graph()

            if "page_messages" not in st.session_state:
                st.session_state.page_messages = {}

            if cls.agent.name not in st.session_state.page_messages:
                st.session_state.page_messages[cls.agent.name] = []

            for message in st.session_state.page_messages[cls.agent.name]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if human_message := st.chat_input():
                if (
                    cls.show_file_uploader
                    and not st.session_state.uploaded_file[cls.agent.name]
                ):
                    st.error(
                        "Please upload a file before sending a message.", icon="🚨"
                    )
                else:
                    cls.stream_events(
                        agent_graph=agent_graph, human_message=human_message
                    )

    @classmethod
    def post_render(cls):
        pass

    @classmethod
    def display(cls):
        cls.pre_render()
        cls.render()
        cls.post_render()
