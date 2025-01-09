import tempfile

import streamlit as st


def add_file_uploader(
    key: str,
    label: str,
    supported_file_types: list[str],
):
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = {}
    if key not in st.session_state.uploaded_file:
        st.session_state.uploaded_file[key] = None

    st.html(
        f"<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>{label}</h3>"
    )

    if uploaded_file := st.file_uploader(
        label=label,
        type=supported_file_types,
        label_visibility="hidden",
    ):
        if not st.session_state["uploaded_file"][key]:
            with tempfile.NamedTemporaryFile(delete=False) as file:
                file.write(uploaded_file.read())
                file.flush()
                st.session_state["uploaded_file"][key] = file.name
