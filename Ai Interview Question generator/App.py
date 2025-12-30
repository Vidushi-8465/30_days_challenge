import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from prompts import generate_prompt

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Page config
st.set_page_config(
    page_title="AI Interview Question Generator",
    page_icon="🎯",
    layout="wide"
)

# Header
st.markdown("""
<h1 style="text-align:center;">🎯 AI Interview Question Generator</h1>
<p style="text-align:center;">
Generate role-specific interview questions based on your skills & job description
</p>
<hr>
""", unsafe_allow_html=True)

# Input layout
col1, col2 = st.columns(2)

with col1:
    skills = st.text_area(
        "🧠 Your Skills",
        placeholder="Python, Machine Learning, SQL, NLP",
        height=150
    )

    difficulty = st.selectbox(
        "🎚 Difficulty Level",
        ["Easy", "Medium", "Hard"]
    )

    q_type = st.selectbox(
        "📌 Question Type",
        ["Conceptual", "Coding", "Scenario-based", "System Design"]
    )

with col2:
    job_desc = st.text_area(
        "🧾 Job Description",
        placeholder="Paste job description here...",
        height=250
    )

generate_btn = st.button("🚀 Generate Interview Questions", use_container_width=True)

# Logic
if generate_btn:
    if not skills or not job_desc:
        st.warning("Please enter skills and job description.")
    else:
        with st.spinner("Generating interview questions..."):
            prompt = generate_prompt(
                skills,
                job_desc,
                difficulty,
                q_type
            )

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )

            output = response.choices[0].message.content

        st.success("Questions Generated Successfully!")

        st.markdown("### 📝 Interview Questions")
        st.markdown(output)

# Footer
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px;">
Built with ❤️ using Streamlit & Groq LLM
</p>
""", unsafe_allow_html=True)
