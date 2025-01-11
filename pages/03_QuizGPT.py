from pydoc import doc
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125"
)

@st.cache_data(show_spinner="Loading files.....")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use",
        ("File", "Wikipedia Article"),
    )
    if choice =="File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf file",
            type=["docx", "txt", "pdf"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia for a topic")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT! This app allows you to generate multiple-choice questions from a text document or a Wikipedia article.
        """
    )

else:
    st.write(docs)
    st.write("Done!")

            