from common.theme import *

page_header_style = f"""
    font-family: {font_family};
    color: {light_beige_color};
    text-align: center;
    font-size: 60px;
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

st.html(f"<h1 style='{page_header_style}'>Learn AI</h1><br>")


def add_banner(banner_title, banner_image_name, banner_content, launch_button_url):
    with container("banner_container"):
        _, image, content, _ = st.columns([0.07, 1, 2, 0.07], gap="medium")
        with image:
            st.image(
                image=f"static/images/{banner_image_name}", use_container_width=True
            )
            st.link_button(
                url=launch_button_url,
                label="**LAUNCH**",
                type="primary",
                use_container_width=True,
            )
        with content:
            st.html(
                f"<h2 style='font-family: {font_family}; color: {coral_color};'>{banner_title}</h2>"
            )

            st.html(
                f"<p style='font-family: {font_family}; font-size: 18px; text-align: justify; color: {light_beige_color}'>{banner_content}</p>"
            )

    st.html("<br/>")


add_banner(
    banner_title="ShopWise Agent",
    banner_image_name="shopwise-agent-banner.jpg",
    banner_content="""
        ShopWise Agent revolutionizes the online shopping experience with the power of agentic AI.
        Whether users have expertise in the product’s field or not, ShopWise Agent helps them discover the best-suited products tailored to their unique needs and preferences. 
        By offering intelligent recommendations and decisive insights, it empowers customers to make informed choices and enjoy a seamless shopping experience.
    """,
    launch_button_url="ShopWise_Agent",
)

add_banner(
    banner_title="Youtube Video Summarizer",
    banner_image_name="youtube-video-summarizer-banner.png",
    banner_content="""
        This agent summarizes YouTube videos using AI-powered natural language processing. 
        Users can input the URL of a YouTube video, and the agent will generate a concise summary of the video's content. 
        The agent uses a combination of video analysis and text summarization to provide a detailed overview of the video. 
        It is a powerful tool for extracting key information from videos and saving time for users.
    """,
    launch_button_url="Youtube_Video_Summarizer",
)

add_banner(
    banner_title="Data Query Agent",
    banner_image_name="data-query-agent-banner.jpeg",
    banner_content="""
        This agent connects natural language queries with data visualization, enabling users to explore datasets effortlessly. 
        Users can upload a SQLite database or a CSV file and ask questions about their data using natural language. 
        The agent translates these questions into SQL queries, executes them on the database, and presents the results as clear and insightful visualizations.
    """,
    launch_button_url="Data_Query_Agent",
)

add_banner(
    banner_title="Financial Analyst Agent",
    banner_image_name="financial-analysis-agent-banner.jpg",
    banner_content="""
        The AI powered financial analyst agent designed to provide insightful and concise analysis to help you make informed financial decisions. 
        Main functions include retrieving and analyzing financial data such as stock prices, historical data, and market trends.
        It uses Yahoo Finance API tools to fetch the data and OpenAI API to generate the insights. 
        It aims to empower you with clear, actionable insights to navigate the financial landscape effectively.
        Users should conduct their own research or consult a financial advisor before making decisions.
    """,
    launch_button_url="Financial_Analyst_Agent",
)

add_banner(
    banner_title="Contextual Retrieval RAG",
    banner_image_name="contextual-retrieval-rag-banner.png",
    banner_content="""
        Contextual retrieval enhances traditional RAG by addressing the issue of insufficient context in individual document chunks.
        Instead of treating chunks as isolated units, it prepends chunk-specific explanatory context to enrich their meaning.
        This approach bridges the gap between granular document splitting and the need for meaningful context in retrieval.
    """,
    launch_button_url="Contextual_Retrieval_RAG",
)

add_banner(
    banner_title="Hybrid Search RAG",
    banner_image_name="hybrid-search-rag-banner.jpg",
    banner_content="""
        Hybrid Search RAG combines lexical search methods like BM25 with semantic search using embeddings.
        This enables both precise keyword matching and deeper understanding of query intent.
        It is ideal for tasks like multi-faceted search, document retrieval, or AI-driven assistance.
        Hybrid Search RAG bridges the gap between traditional search techniques and modern semantic capabilities.
    """,
    launch_button_url="Hybrid_Search_RAG",
)

add_banner(
    banner_title="Graph RAG",
    banner_image_name="graph-rag-banner.jpg",
    banner_content="""
        Graph RAG organizes data as nodes and edges in a graph, capturing relationships between concepts.
        This enables more context-aware and accurate responses compared to traditional RAG.
        It is ideal for tasks like complex reasoning, exploring knowledge connections, or semantic research.
        Graph RAG bridges the gap between relational data and AI-driven, knowledge-based insights.
    """,
    launch_button_url="Graph_RAG",
)

add_banner(
    banner_title="Demystify RAG",
    banner_image_name="demystify-rag-banner.jpg",
    banner_content="""
        This app gives you a visual representation of how RAG works.
        It takes you to the tour of entire steps involved in RAG.
        It begins with uploading documents, which are then divided into smaller chunks and indexed using embeddings for efficient retrieval.
        When a query is received, the system performs a similarity search to fetch the most relevant information.
        These retrieved chunks are passed to a generative AI model to create accurate, context-aware responses.
    """,
    launch_button_url="Demystify_RAG",
)

add_banner(
    banner_title="Simple RAG",
    banner_image_name="simple-rag-banner.jpg",
    banner_content="""
        One of the most common use cases of Generative AI is RAG.
        RAG involves retrieving relevant information from a knowledge base and incorporating it into the generation process.
        It is ideal for tasks like document retrieval, multi-faceted search, or AI-driven assistance.
        RAG is effective for tasks like document retrieval, multi-faceted search, or AI-driven assistance.
    """,
    launch_button_url="Simple_RAG",
)
