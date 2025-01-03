import streamlit as st
from langchain.storage import LocalFileStore
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
#     # st.session_state.messages = []
# =========================================
#  This code doesnt need anymore because initialized in the last line
# =========================================

class ChatCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs,):
        with st.sidebar:
            st.write("AI is Started...")
    
    def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs):
        with st.sidebar:
            st.write("AI is Ended...")

    def on_llm_new_token(self, token, *, chunk = None, run_id, parent_run_id = None, **kwargs):
        print(token)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ],
    )

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="💬"
)

@st.cache_data(show_spinner="Embedding file...")
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

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state.messages.append({"message": message, "role": role})

def paint_history ():
    for message in st.session_state.messages:
    # for message in st.session_state["messages"]: is same with above
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the question using ONLY the following context. If you don't know the nswer just say you dont know. Don't make anything up.

    Context: {context}
    """),
    ("human", "{question}"),]
)

st.title("DocumentGPT")


st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    """
    )

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf","txt","docx"],)

if file:   
    retriever = embed_file(file)

    send_message("I'm ready!! ask me anything about your file", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask anything!!")

    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm

        responses = chain.invoke(message)
        send_message(responses.content, "ai")
        
        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # # 위의코드는 각 도큐먼트 (여기에서는 4개의 리트리버)중간에 줄바꿈을 두번 해서 하나로 합친다는 코드
        # prompt = template.format_messages(context=docs, question=message)
        # llm.predict_messages(prompt)
else:
    st.session_state["messages"] = []
    # if there was no file upload, we need to reset the messages
