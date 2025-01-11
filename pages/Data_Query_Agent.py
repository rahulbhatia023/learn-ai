import time

import pandas as pd
import streamlit as st

from agents.data_query_agent import DataQueryAgent
from common.chat import add_chat_message, display_message
from common.file_uploader import add_file_uploader, is_file_uploaded
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

    agent_graph = agent.get_graph()
    add_langgraph_workflow_visualization(agent_graph)

if "page_messages" not in st.session_state:
    st.session_state.page_messages = {}

if agent.agent_name not in st.session_state.page_messages:
    st.session_state.page_messages[agent.agent_name] = []

if not keys_missing(agent.required_api_keys):
    if query := st.chat_input(placeholder="Write your query here..."):
        if is_file_uploaded(key=agent.agent_name):
            config = {"configurable": {"thread_id": "1"}}

            agent_input = {"question": query}

            add_chat_message(agent_name=agent.agent_name, role="human", content=query)

            with st.spinner(text="Thinking..."):
                result = agent_graph.invoke(
                    input=agent_input,
                    config=config,
                )

            def stream_data(content):
                for word in content.split(" "):
                    yield word + " "
                    time.sleep(0.04)

            if result:
                with st.chat_message("ai"):
                    if result["results"] == "NOT_RELEVANT":
                        st.write_stream(
                            stream_data(
                                content="I'm sorry, I couldn't find any relevant data."
                            )
                        )
                    else:
                        st.write_stream(stream_data(content=result["answer"]))

                        st.subheader("SQL Query:")
                        st.code(
                            body=result["sql_query"], language="sql", wrap_lines=True
                        )

                        dataframe = pd.DataFrame(
                            data=result["results"], columns=result["query_columns"]
                        )

                        st.subheader("Data:")
                        st.dataframe(
                            data=dataframe,
                            hide_index=True,
                        )

                        st.subheader("Data Visualization:")
                        st.bar_chart(
                            data=dataframe,
                            x=dataframe.columns[0],
                            y=dataframe.columns[1],
                            x_label=dataframe.columns[0],
                            y_label=dataframe.columns[1],
                        )
