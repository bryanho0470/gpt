import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    temperature=0.1
)

answer_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. IF you can't just say you don't know, don't make naything up.

    Then, give a score to the answer between and 5. 0 being not helpful to the user and 5 being helpful to the user.

    You don't need to show the score in final answer.

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    your turn!

    Context: {context}
    Question: {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Using the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources. Return the score as it is.
            And you need to conversation like as a owner the site.

            Answer: {answers}
            """,
        ),
        (
            "human", "{question}"
        ),
    ]
)

def get_answers(inputs):
    docs = inputs['doc']
    question = inputs['question']
    answer_chain = answer_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answer_chain.invoke({
    #         "question": question,
    #         "context" : doc.page_content
    #     })
    #     answers.append(result.content)
    return {
        "question" : question,
        "answers" :[
            {
            "answer": answer_chain.invoke({
                "question": question, "context": doc.page_content
            }).content,
            "source": doc.metadata["source"],
            
            } for doc in docs
        ],
    }

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\n" for answer in answers)
    return choose_chain.invoke({
        "question":question,
        "answers": condensed,
    })






def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
        # decompose is to delet include header tag in teh soup
    if footer:
        footer.decompose()
    return (
         str(soup.get_text()).replace("\n", " ").replace("\u3000", " ").replace("\t"," ").replace("nNEXT\nNEWS", " ")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(
        text_splitter=splitter
    )
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

st.set_page_config(
    page_title="SiteGPT",
    page_icon="📺"
)

st.title("SiteGPT")

st.markdown(
    """
    
    #Site GPT
    
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.

    """)

with st.sidebar:
        url = st.text_input(
                "Write down a URL",
                placeholder="https://example.com",
        )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input("You can ask anything about this WEBSITE")
        if query:
            chain = {"doc": retriever, "question" : RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)

            st.write(result.content)
