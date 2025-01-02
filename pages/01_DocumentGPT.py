import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="💬"
)

st.title("DocumentGPT")


st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    """
    )

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf","txt","docx"],)

if file:
    st.write(file)
    file_content = file.read()
    file_path=f"./.cache/files/{file.name}"
    st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)


    