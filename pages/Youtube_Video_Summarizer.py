import streamlit as st

from agents.youtube_video_summarizer_agent import YoutubeVideoSummarizerAgent
from common.langgraph import add_langgraph_workflow_visualization
from common.page import get_api_key, keys_missing
from common.theme import set_page_config

agent = YoutubeVideoSummarizerAgent

set_page_config(page_title=agent.agent_name)

with st.sidebar:
    # API KEYS

    get_api_key(agent.required_api_keys)

    st.divider()

    # LANGGRAPH WORKFLOW VISUALIZATION

    agent_graph = agent.get_graph()
    add_langgraph_workflow_visualization(agent_graph)

if not keys_missing(agent.required_api_keys):
    if url := st.chat_input(placeholder="Enter the YouTube video URL"):
        st.video(url)

        config = {"configurable": {"thread_id": "1"}}

        agent_input = {"video_url": url}

        with st.empty():
            with st.spinner(text="Thinking..."):
                state = agent_graph.invoke(
                    input=agent_input,
                    config=config,
                )

            if state:
                st.write(state)
