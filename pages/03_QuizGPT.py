import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],
)

def format_docs(docs):
    """Format retrieved documents."""
    return "\n\n".join(document.page_content for document in docs)

question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """ You are a helpful assistant that is role playing as a teacher. Based ONLY on the following context make 10 questions to test the user's knowledge about athe text.
            Each question should have 4 answers, three of them must be incorrect and one must be correct.
            Use (o) to signal the correct answer.
            
            Question example: 
            Question: What is the color of the ocean?
            Answers: Red|Yello|Green|Blue(o)

            Question: What isthe capital of Georgia?
            Answers: Baku | Tbilisi(o) | Yerevan | Ankara

            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998

            Question: Who was Julius Caesar?
            Answers: A roman Emperor (o) | Painter | Actor | Model
            
            Your turn!

            context : {context}
            

            """),
        ]
    )

question_chain = {"context": format_docs} | question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a powerfil formatting algorithm.
        
        You format exam questions into JSON format.
        Answers with (o) are the correct ones.

        Example Imput:

        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
             
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
             
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model

        Example Output:
        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow",
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }},
                    ]
                }},
                            {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi",
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }},
                    ]
                }},
                            {{
                    "question": "When was Avatar released?",
                    "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001",
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }},
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter",
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }},
                    ]
                }}
            ]
        }}
        ```
         Your turn!

         Question : {context}

         """
         )
    ]
)

formatting_chain = formatting_prompt | llm


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


    start = st.button("Generate your Quiz")

    if start:
        question_response = question_chain.invoke(docs)
        st.write(question_response.content)
        formatting_response = formatting_chain.invoke(
            {"context": question_response.content}
        )
        st.write(formatting_response.content)

            