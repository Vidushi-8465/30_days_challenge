import streamlit as st
import os
from groq import Groq

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Text Summarizer",
    layout="centered"
)

st.title(" AI Text Summarizer")
st.caption("Summarize long text using Groq LLMs")

# ---------------- API CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("Please set your GROQ_API_KEY as an environment variable.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------- UI ----------------
input_text = st.text_area(
    "Enter text to summarize",
    height=260,
    placeholder="Paste your article, notes, or document here..."
)

summary_length = st.selectbox(
    "Summary Length",
    ["Short", "Medium", "Detailed"]
)

summarize_btn = st.button("Generate Summary")

# ---------------- Prompt Builder ----------------
def build_prompt(text, length):
    if length == "Short":
        instruction = "Summarize the text in 3–4 concise bullet points."
    elif length == "Medium":
        instruction = "Summarize the text in a clear and concise paragraph."
    else:
        instruction = "Provide a detailed yet concise summary covering all key points."

    return f"{instruction}\n\nText:\n{text}"

# ---------------- LLM Call ----------------
def generate_summary(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful text summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )

    return response.choices[0].message.content

# ---------------- Logic ----------------
if summarize_btn:
    if not input_text.strip():
        st.error("Please enter some text to summarize.")
    elif not GROQ_API_KEY:
        st.error("" \
        "" \
        "Groq API key not found.")
    else:
        with st.spinner("Summarizing..."):
            try:
                prompt = build_prompt(input_text, summary_length)
                summary = generate_summary(prompt)

                st.subheader("Summary")
                st.success(summary)

            except Exception as e:
                st.error("Something went wrong while generating the summary.")
                st.exception(e)
