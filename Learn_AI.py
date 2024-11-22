import streamlit as st

page_header = "Learn AI"

font_family = "Poppins"

dark_green_color = "#16423C"
mild_green_color = "#6A9C89"
light_green_color = "#C4DAD2"

page_header_style = f"""
    font-family: {font_family};
    color: {light_green_color};
    text-align: center;
    font-size: 60px;
"""

page_subheader_style = f"""
    font-family: {font_family};
    color: {light_green_color};
    text-align: center;
    font-size: 20px;
"""

card_title_style = f"""
    font-family: {font_family};
    color: {light_green_color};
    text-align: center;
    font-size: 30px;
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
        }
    </style>
    """
)

st.html(f"<h1 style='{page_header_style}'>{page_header}</h1>")

container = st.container()

col11, col12, _ = container.columns([1, 1, 1], gap="large")

with col11:
    with st.container(border=True):
        st.html(f"<h3 style='{card_title_style}'>Document Chunking</h1>")
        st.image(image="static/images/simple-rag-banner.jpg", use_container_width=True)
        st.html(
            f"""
                    <p style='font-family: {font_family};text-align: center'>
                        <ul>
                            One of the most common use cases of Generative AI is RAG (Retrieval Augmented Generation).
                            RAG applications are tools that combine AI language models with real-world information sources to give better answers. 
                            They fetch data from documents, websites, or databases and then generate answers using AI. 
                            This makes them more accurate and up-to-date compared to regular AI models that rely only on training data. 
                            RAG is great for tasks like answering questions, summarizing documents, or helping with research. 
                            It bridges the gap between advanced AI and real-time, fact-based knowledge.
                        </ul>
                    </p>
                """
        )
        st.link_button(
            url="Document_Chunking",
            label="**LAUNCH**",
            type="primary",
            use_container_width=True,
        )

with col12:
    with st.container(border=True):
        st.html(f"<h3 style='{card_title_style}'>Simple RAG</h1>")
        st.image(image="static/images/simple-rag-banner.jpg", use_container_width=True)
        st.html(
            f"""
                    <p style='font-family: {font_family};text-align: center'>
                        <ul>
                            One of the most common use cases of Generative AI is RAG (Retrieval Augmented Generation).
                            RAG applications are tools that combine AI language models with real-world information sources to give better answers. 
                            They fetch data from documents, websites, or databases and then generate answers using AI. 
                            This makes them more accurate and up-to-date compared to regular AI models that rely only on training data. 
                            RAG is great for tasks like answering questions, summarizing documents, or helping with research. 
                            It bridges the gap between advanced AI and real-time, fact-based knowledge.
                        </ul>
                    </p>
                """
        )
        st.link_button(
            url="Simple_RAG",
            label="**LAUNCH**",
            type="primary",
            use_container_width=True,
        )
