import streamlit as st

page_header = "Learn AI"
page_subheader = "AI Simplified, Skills Amplified"

font_family = "Poppins"

dark_green_color = "#16423C"
mild_green_color = "#6A9C89"
light_green_color = "#C4DAD2"

page_header_style = f"""
    font-family: {font_family};
    color: {light_green_color};
    text-align: center;
"""

page_subheader_style = f"""
    font-family: {font_family};
    color: {light_green_color};
    text-align: center;
    font-size: 15px;
"""

vertical_space_div_style = f"""
    margin-top: 80px;
    margin-bottom: 80px;
"""

st.set_page_config(
    page_title="Learn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins');
    
    .stApp {
        background-color: #16423C;
        font-family: 'Poppins';
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='{page_header_style}'>{page_header}</h1>", unsafe_allow_html=True
)

st.markdown(
    f"<h3 style='{page_subheader_style}'>{page_subheader}</h3>", unsafe_allow_html=True
)

st.markdown(f"<div style='{vertical_space_div_style}' />", unsafe_allow_html=True)

with st.container(border=True, key="simple-rag-container"):
    st.page_link(
        page="pages/Simple_RAG.py",
        label="Simple RAG",
        icon="🤖",
        use_container_width=False,
    )
