from streamlit_extras.stylable_container import stylable_container
import streamlit as st

font_family = "Poppins"
coral_color = "#FF6F61"
light_beige_color = "#F5F5DC"

app_title_style = f"""
    '
        font-family: {font_family};
        color: {coral_color}; 
        text-align: center;
    '
"""

app_container_title_style = f"""
    '
        font-family: {font_family};
        color: {light_beige_color}; 
        text-align: center;
    '
"""

app_container_style = """
    {
        background-color: #194d46;
        border-radius: 20px;
        border: 1px solid white;
        padding-block: 30px;
    }                   
"""


def set_page_config(
    page_title: str,
):
    st.set_page_config(
        page_title=page_title,
        page_icon="🤖",
        layout="wide",
    )

    st.html(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins');

            .stApp {
                font-family: 'Poppins';
                background-color: #16423C;
            }

            [title="Show password text"] {
                display: none;
            }
        </style>
        """
    )

    st.html(f"<h1 style={app_title_style}>{page_title}</h1><br/>")


def container(key):
    return stylable_container(
        key=key,
        css_styles=app_container_style,
    )


def container_title(title):
    st.html(f"<h3 style={app_container_title_style}>{title}</h3>")
