from operator import itemgetter
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
#     # st.session_state.messages = []
# =========================================
#  This code doesnt need anymore because initialized in the last line
# =========================================

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, serialized, prompts, *args, **kwargs,):
        self.message_box = st.empty()
    
    def on_llm_end(self, response, *args, **kwargs):
        save_message(self.message, "ai")    

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        # same with self.message = f'{self.message}{token}'
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="💬"
)

def format_docs(docs):
    """Format retrieved documents."""
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Embedding file...")
def embed_file (file):
    file_content = file.read()
    file_path=f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = embeddings
    vectorstore = FAISS.from_documents(docs, cached_embeddings) # we need to find another vectorstore service for further implementation. Chroma and FAISS is free vectorstore
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    """save the message to the session state"""
    st.session_state.messages.append({"message": message, "role": role})

def save_memory(input, output):
    """Save the memory to the session state"""
    st.session_state["history"].append({"input":input, "output" : output})

def send_message(message, role, save=True):
    """Send and Display messafges in the chat ingterface"""
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)
       
def paint_history ():
    """Replay chat history"""
    for message in st.session_state.messages:
    # for message in st.session_state["messages"]: is same with above
        send_message(message["message"], message["role"], save=False)

def restore_memory ():
    """Restore memory from the session state"""
    for history in st.session_state["history"]:
        st.session_state["memory"].save_context({"input":history["input"]}, {"output":history["output"]})

def invoke_chain(message):
    result = chain.invoke(message)
    save_memory(message, result.content)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the question using ONLY the following context. If you don't know the nswer just say you dont know. Don't make anything up.

    Context: {context}
     
    And you will get about summarized context of previouse chat history below. If It is empty, you don't need to care about chat history yet.
    Chat History: {history}
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
    st.subheader("API setting")
    openai_api_key = st.text_input("Enter your OpenAI API KEY!")

memory_llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key
)
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ],
    )



if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm,
        max_token_limit=100,
        memory_key="history",
        return_messages=True,
    )


if file:   
    retriever = embed_file(file)

    send_message("I'm ready!! ask me anything about your file", "ai", save=False)
    restore_memory()
    paint_history()

    message = st.chat_input("Ask anything!")

    if message:

        send_message(message, "human")
      
        chain = ({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(history=RunnableLambda(st.session_state["memory"].load_memory_variables) | itemgetter("history")        
            )
            # RunnablePassthrough and RunnableLambda is important to implement the chain. Passthrought is just pass the value to next chain and Lambda is to run the function in the chain 
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            try:
                invoke_chain(message)
            except Exception as e:
                st.write(e)
        # send_message(responses.content, "ai")
        
        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # # 위의코드는 각 도큐먼트 (여기에서는 4개의 리트리버)중간에 줄바꿈을 두번 해서 하나로 합친다는 코드
        # prompt = template.format_messages(context=docs, question=message)
        # llm.predict_messages(prompt)
else:
    st.session_state["messages"] = []
    st.session_state["history"] = []
    # if there was no file upload, we need to reset the messages
