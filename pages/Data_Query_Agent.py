import tempfile

import streamlit as st

from agents.data_query_agent import DataQueryAgent
from common.file_uploader import add_file_uploader
from common.langgraph import add_langgraph_workflow_visualization
from common.page import get_api_key, keys_missing
from common.theme import set_page_config

agent = DataQueryAgent

set_page_config(page_title=agent.agent_name)

with st.sidebar:
    # API KEYS

    get_api_key(agent.required_api_keys)

    st.divider()

    # FILE UPLOADER

    add_file_uploader(
        key=agent.agent_name,
        label="Upload SQLite File",
        supported_file_types=["sqlite"],
    )

    st.divider()

    # LANGGRAPH WORKFLOW VISUALIZATION

    add_langgraph_workflow_visualization(agent.get_graph())

if not keys_missing(agent.required_api_keys):
    if query := st.chat_input(placeholder="Write your query here..."):
        st.write(query)
