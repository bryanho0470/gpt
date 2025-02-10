import streamlit as st
import subprocess
from pydub import AudioSegment
import openai
import math
import glob

def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3").replace("mkv","mp3").replace("avi","mp3").replace("mov","mp3")
    command = ["ffmpeg", "-i", video_path, "-vn", audio_path,]

    return subprocess.run(command)

def cut_audio_to_chunks(video_path,chunk_size, chunks_folder):
    audio_path = video_path.replace("mp4", "mp3").replace("mkv","mp3").replace("avi","mp3").replace("mov","mp3")
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i+1) * chunk_len

        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")

def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3")
    for file in files:
        with open(file, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                "whisper-1", audio_file
            )
        with open(destination, "a") as text_file:
            text_file.write(transcript["text"])

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📅"
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi","mkv","mov",])
    selected_chunk_size = st.selectbox("Select Chunk Size (min)",[3,5,10], index=1)
    chunks_folder = "./files/chunks"
    destination = "./files/transcripts/final_transcript.txt"
    if video:
        video_content = video.read()
        video_path = f"./.cache/mp3/{video.name}"
        with open(video_path, "wb") as f:
            f.write(video_content)
        extract_audio_from_video(video_path)
        cut_audio_to_chunks(video_path, selected_chunk_size, chunks_folder)
        transcribe_chunks(chunks_folder, destination)




st.title("MeetingGPT")

