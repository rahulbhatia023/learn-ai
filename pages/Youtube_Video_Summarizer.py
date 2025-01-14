import time

import streamlit as st
from pytubefix import YouTube

from agents.youtube_video_summarizer_agent import YoutubeVideoSummarizerAgent
from common.langgraph import add_langgraph_workflow_visualization
from common.page import get_api_key, keys_missing
from common.theme import (
    set_page_config,
    app_container_title_style,
    container,
    container_title,
)

agent = YoutubeVideoSummarizerAgent

set_page_config(page_title=agent.agent_name, page_layout="centered")

with st.sidebar:
    # API KEYS

    get_api_key(agent.required_api_keys)

    st.divider()

    # LANGGRAPH WORKFLOW VISUALIZATION

    agent_graph = agent.get_graph()
    add_langgraph_workflow_visualization(agent_graph)

if not keys_missing(agent.required_api_keys):
    if url := st.chat_input(placeholder="Enter the YouTube video URL"):
        # Video Preview
        st.video(url)

        # Video Title
        video = YouTube(url)
        st.html(f"<h1 style={app_container_title_style}>{video.title}</h1>")

        with container("video_summary"):
            container_title("Summary")

            _, col2, _ = st.columns([0.5, 8, 0.5])

            with col2:
                with st.empty():
                    with st.spinner(text="Analyzing..."):
                        state = agent_graph.invoke(
                            input={"video_url": url},
                            config={"configurable": {"thread_id": "1"}},
                        )
                    if state:

                        def stream_data(content):
                            for word in content.split(" "):
                                yield word + " "
                                time.sleep(0.04)

                        st.write_stream(stream_data(state["summary"]))
