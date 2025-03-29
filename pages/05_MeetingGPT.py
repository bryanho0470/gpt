import streamlit as st
import subprocess #to make a code sa same like CLI
from pydub import AudioSegment #to make object can be search from audio file. 
import openai #make transcript with Whisper from audio file
import math # calculate 
import glob #for searching specific file in the folder return object.
import os #for control files
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📅"
)

with st.sidebar:
    st.subheader("API setting")

    if 'api_confirmed' not in st.session_state:
        st.session_state.api_confirmed = False

    if not st.session_state["api_confirmed"]:
        openai_api_key = st.text_input("Enter your OpenAI API KEY!", type="password")
        confirm_button = st.button("Confirm API Key")

        if confirm_button:
            if openai_api_key and openai_api_key.startswith("sk-"):
                st.session_state["api_key"] = openai_api_key
                st.session_state["api_confirmed"] = True
                st.success("API key confirmed!")
            else:
                st.error("Invalid API key.")
                st.stop()
        else:
            st.stop()
    else:
        openai_api_key = st.session_state["api_key"]
        st.success("API key confirmed!")
        st.balloons()

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
)


has_transcript = os.path.exists("./files/transcripts/final_transcript.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=100,)

@st.cache_resource()
def embed_file (file_path):
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings) 
    retriever = vectorstore.as_retriever()
    return retriever

@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3").replace("mkv","mp3").replace("avi","mp3").replace("mov","mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path,]
    return subprocess.run(command)

@st.cache_data()
def cut_audio_to_chunks(video_path,chunk_size, chunks_folder):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3").replace("mkv","mp3").replace("avi","mp3").replace("mov","mp3")
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i+1) * chunk_len

        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                "whisper-1", audio_file
            )
        with open(destination, "a") as text_file:
            text_file.write(transcript["text"])



st.markdown(f"already has transcript? =  {has_transcript}")

st.title("MeetingGPT")
st.markdown("""
    Welcome to Meeting GPT. Please upload your recoreded meeting Video.
    Then, we will provide a transcript and summarized minutes.

""")

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi","mkv","mov",])
    selected_chunk_len = st.selectbox("Select Chunk Size (min)",[3,5,10], index=1)
    chunks_folder = "./files/chunks"
    destination = "./files/transcripts/final_transcript.txt"

if video:
    with st.status("Uplading Video now...") as status:
        video_content = video.read()
        video_path = f"./.cache/mp3/{video.name}"
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extract audio segment now...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting Audio for preparing using whisper")
        cut_audio_to_chunks(video_path, selected_chunk_len, chunks_folder)
        status.update(label="Making whole Transcript..Please wait..")
        transcribe_chunks(chunks_folder, destination)
        status.update(label="All done! Thank you for your patients")
        st.write("🎉")

transcript_tab, summary_tab, qna_tab = st.tabs(["Transcript","Summary","Q&A",])

with transcript_tab:
    with open(destination) as file:
        st.write(file.read())

with summary_tab:
    start = st.button("Generate Summary")
    if start:
        loader = TextLoader(destination)
        docs = loader.load_and_split(text_splitter=splitter)
        first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        Concise summary:
        """
        )
        first_summary_chain = first_summary_prompt | llm | StrOutputParser()

        summary = first_summary_chain.invoke({
            "text":docs[0].page_content
        })

        refine_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to produce a final summary.
            We have provided an existing summary up to a certain point: {exsiting_summary}
            We have the opportunity to refine the existing summary (only if needed) with some more context belo.
            ----------
            {context}
            ----------
            Given the new context, refine the original summary.
            If the context isn't useful, RETURN the original summary.            
            """
        )

        refine_chain = refine_prompt | llm | StrOutputParser()

        with st.status("Summarizing......") as status:
            for i, doc in enumerate(docs[1:]):
                status.update(label=f"Processing Document {i+1}/{len(docs)-1}")
                summary = refine_chain.invoke({
                    "exsiting_summary": summary,
                    "context": doc.page_content
                })
                st.write(summary)
        st.write(summary)

with qna_tab:

    retriever = embed_file(destination)
    docs = retriever.invoke("Do They talk about country that it is related to Silk road?")
    st.write(docs)





