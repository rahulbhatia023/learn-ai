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

graph_rag, demystify_rag, simple_rag = st.columns([1, 1, 1], gap="large")

with graph_rag:
    with st.container(border=True):
        st.html(f"<h3 style='{card_title_style}'>Graph RAG</h1>")
        st.image(image="static/images/graph-rag-banner.jpg", use_container_width=True)
        st.html(
            f"""
                    <p style='font-family: {font_family};text-align: center'>
                            One of the most common use cases of Generative AI is RAG.
                            RAG applications fetch data from documents, websites, or databases and then generate answers using AI. 
                            This makes them more accurate and up-to-date compared to regular AI models that rely only on training data. 
                            RAG is great for tasks like answering questions, summarizing documents, or helping with research. 
                            It bridges the gap between advanced AI and real-time, fact-based knowledge.
                    </p>
                """
        )
        st.link_button(
            url="Graph_RAG",
            label="**LAUNCH**",
            type="primary",
            use_container_width=True,
        )

with demystify_rag:
    with st.container(border=True):
        st.html(f"<h3 style='{card_title_style}'>Demystify RAG</h1>")
        st.image(
            image="static/images/demystify-rag-banner.jpg", use_container_width=True
        )
        st.html(
            f"""
                    <p style='font-family: {font_family};text-align: center'>
                            This app gives you a visual representation of how RAG works.
                            It takes you to the tour of entire steps involved in RAG.
                            It begins with uploading documents, which are then divided into smaller chunks and indexed using embeddings for efficient retrieval.
                            When a query is received, the system performs a similarity search to fetch the most relevant information. 
                            These retrieved chunks are passed to a generative AI model to create accurate, context-aware responses.
                    </p>
                """
        )
        st.link_button(
            url="Demystify_RAG",
            label="**LAUNCH**",
            type="primary",
            use_container_width=True,
        )

with simple_rag:
    with st.container(border=True):
        st.html(f"<h3 style='{card_title_style}'>Simple RAG</h1>")
        st.image(image="static/images/simple-rag-banner.jpg", use_container_width=True)
        st.html(
            f"""
                    <p style='font-family: {font_family};text-align: center'>
                            One of the most common use cases of Generative AI is RAG.
                            RAG applications fetch data from documents, websites, or databases and then generate answers using AI. 
                            This makes them more accurate and up-to-date compared to regular AI models that rely only on training data. 
                            RAG is great for tasks like answering questions, summarizing documents, or helping with research. 
                            It bridges the gap between advanced AI and real-time, fact-based knowledge.
                    </p>
                """
        )
        st.link_button(
            url="Simple_RAG",
            label="**LAUNCH**",
            type="primary",
            use_container_width=True,
        )
