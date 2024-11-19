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

st.set_page_config(
    page_title="Learn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins');
        
        .stApp {
            background-color: #16423C;
            font-family: 'Poppins';
        }
        
        [data-testid="stImage"] {
            border-radius: 10px;
            overflow: hidden;
            margin-left: 20px;
        }
    </style>
    """
)

st.html(f"<h1 style='{page_header_style}'>{page_header}</h1>")
st.html(f"<h3 style='{page_subheader_style}'>{page_subheader}</h3><br>")

simple_rag_container = st.container(border=True, key="simple-rag-container")

with simple_rag_container:
    st.html(f"<h1 style='{page_header_style}'>Simple RAG</h1><br>")

    col1, col2 = simple_rag_container.columns(
        spec=[1, 2], gap="large", vertical_alignment="center"
    )

    with col1:
        st.image(image="static/images/simple-rag-banner.jpg")

    with col2:
        st.html(
            f"""
                <p style='font-family: {font_family};'>
                    One of the most common use cases of Generative AI is RAG (Retrieval Augmented Generation).
                    <br><br> RAG applications are tools that combine AI language models with real-world information sources to give better answers. 
                    <br><br> They fetch data from documents, websites, or databases and then generate answers using AI. 
                    <br><br> This makes them more accurate and up-to-date compared to regular AI models that rely only on training data. 
                    <br><br> RAG is great for tasks like answering questions, summarizing documents, or helping with research. 
                    <br><br> It bridges the gap between advanced AI and real-time, fact-based knowledge.
                </p>
            """
        )

    _, col2, _ = simple_rag_container.columns([1, 1, 1])

    with col2:
        st.html("<br>")
        st.link_button(
            url="Simple_RAG", label="Launch", type="primary", use_container_width=True
        )
        st.html("<br>")
