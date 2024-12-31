import time

import streamlit as st


def add_chat_message(agent_name: str, role: str, content: str):
    st.session_state.page_messages[agent_name].append(
        {"role": role, "content": content}
    )

    def stream_data():
        for word in content.split(" "):
            yield word + " "
            time.sleep(0.04)

    with st.chat_message(role):
        if role == "ai":
            st.write_stream(stream_data)
        else:
            st.write(content)


def display_message(agent_name: str, v):
    if "messages" in v:
        m = v["messages"][-1]
        if (m.type == "ai" and not m.tool_calls) or m.type == "human":
            add_chat_message(agent_name=agent_name, role=m.type, content=m.content)
