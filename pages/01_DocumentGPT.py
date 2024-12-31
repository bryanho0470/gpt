import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="💬"
)

st.title("DocumentGPT")

with st.chat_message("Human"):
    st.write("Who are you?")

with st.chat_message("ai"):
    st.write("how are you?")

st.chat_input("Send message to AI")

with st.status("Embedding Files....", expanded=True) as status:
    time.sleep(3)
    st.write("Getting the file")
    time.sleep(3)
    st.write("Embedding the file")
    time.sleep(3)
    st.write("Caching the file")
    status.update(label="Error", state="error")
    