import streamlit as st


def add_chat_message(agent_name: str, role: str, content: str):
    st.session_state.page_messages[agent_name].append(
        {"role": role, "content": content}
    )
    with st.chat_message(role):
        st.markdown(f"<p class='fontStyle'>{content}</p>", unsafe_allow_html=True)


def display_message(agent_name: str, v):
    if "messages" in v:
        m = v["messages"][-1]
        if (m.type == "ai" and not m.tool_calls) or m.type == "human":
            add_chat_message(agent_name=agent_name, role=m.type, content=m.content)
