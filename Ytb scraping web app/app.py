import streamlit as st
import subprocess
import os
import requests
from bs4 import BeautifulSoup
import tempfile
import whisper

# ================= CONFIG =================
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "llama3-8b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ================= GROQ LLM =================
def summarize_text(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional summarizer."
            },
            {
                "role": "user",
                "content": f"Summarize the following content in clear bullet points:\n\n{text}"
            }
        ],
        "temperature": 0.3
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# ================= YOUTUBE TRANSCRIPT =================
def get_youtube_transcript(url):
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "-o", f"{tmpdir}/video",
            url
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for file in os.listdir(tmpdir):
            if file.endswith(".vtt"):
                with open(os.path.join(tmpdir, file), "r", encoding="utf-8") as f:
                    return f.read()

    return None

# ================= AUDIO FALLBACK =================
def audio_to_text(url):
    model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, url])
        result = model.transcribe(audio_path)
        return result["text"]

# ================= WEB SCRAPER =================
def scrape_website(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    paragraphs = [p.text for p in soup.find_all("p")]
    return " ".join(paragraphs)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="YT / Web Summarizer", layout="wide")
st.title("🎯 YouTube & Web Scrape Summarizer (Groq LLM)")

option = st.radio("Choose Input Type:", ["YouTube Video", "Website URL"])

url = st.text_input("Enter URL")

if st.button("Summarize"):
    with st.spinner("Processing..."):
        if option == "YouTube Video":
            text = get_youtube_transcript(url)

            if not text:
                st.warning("No captions found. Using audio transcription...")
                text = audio_to_text(url)

        else:
            text = scrape_website(url)

        if text:
            summary = summarize_text(text[:12000])
            st.subheader("📌 Summary")
            st.write(summary)
        else:
            st.error("Could not extract text.")
