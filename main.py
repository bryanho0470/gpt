import streamlit as st
from langchain.prompts import PromptTemplate

st.title("hello world!")

st.subheader("Welcome to Streamlit!!")

st.markdown(
    """
    #### I love it!
    """
)

st.write("hi")

st.write([1,2,3,4])

st.write(PromptTemplate)

p = PromptTemplate.from_template("xxx")

st.write(p)

st.selectbox(
    label="Choose you model",
    options=("GPT-3","GPT-4"),
    placeholder="Please select"
    )
