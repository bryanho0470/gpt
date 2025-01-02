from curses import qiflush
import streamlit as st
from langchain.storage import LocalFileStore
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS, Chroma



st.set_page_config(
    page_title="DocumentGPT",
    page_icon="💬"
)

def embed_file (file):
    file_content = file.read()
    file_path=f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore("./.cache/embeddings")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings) # we need to find another vectorstore service for further implementation. Chroma and FAISS is free vectorstore
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")


st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    """
    )

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf","txt","docx"],)

if file:   
    retriever = embed_file(file)
    answer = retriever.invoke("Supporter")
    answer   