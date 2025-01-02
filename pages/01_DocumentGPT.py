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

    cache_dir = LocalFileStore("./.cache/embeddings")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/allotment.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings) # we need to find another vectorstore service for further implementation. Chroma and FAISS is free vectorstore
    retriever = vectorstore.as_retriever()
    docs = retriever.index("allotment of shares")
    st.write(docs)    