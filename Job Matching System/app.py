import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Job Matching & ATS System",
    page_icon="🤖",
    layout="wide"
)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- DATA ----------------
SKILLS = [
    "python","java","javascript","sql","html","css","react","node",
    "machine learning","deep learning","nlp","data analysis",
    "docker","kubernetes","aws","cloud","devops","linux",
    "cybersecurity","api","rest","flask","django","streamlit"
]

JOB_ROLES = {
    "Machine Learning Engineer": ["python","machine learning","deep learning","nlp"],
    "Data Scientist": ["python","sql","data analysis","machine learning"],
    "Backend Developer": ["python","java","api","django","flask"],
    "Frontend Developer": ["html","css","javascript","react"],
    "DevOps Engineer": ["docker","kubernetes","aws","linux","devops"],
    "Cybersecurity Analyst": ["cybersecurity","linux","network","security"]
}

# ---------------- HELPERS ----------------
def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    return " ".join([p.extract_text() for p in reader.pages])

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())

def extract_skills(text):
    return list(set(skill for skill in SKILLS if skill in text))

# ---------------- UI ----------------
st.markdown("""
<h1 style='text-align:center;'>🤖 AI Job Matching & ATS System</h1>
<p style='text-align:center;'>Resume Intelligence • ATS Scoring • Career Recommendation</p>
<hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    resume_pdf = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    job_desc = st.text_area("🧾 Paste Job Description", height=300)

analyze = st.button("🔍 Analyze Resume", use_container_width=True)

# ---------------- LOGIC ----------------
if analyze:
    if not resume_pdf or not job_desc:
        st.warning("Upload resume and job description.")
    else:
        with st.spinner("AI analyzing..."):
            resume_text = clean_text(extract_text_from_pdf(resume_pdf))
            job_text = clean_text(job_desc)

            # Embeddings
            r_emb = model.encode([resume_text])
            j_emb = model.encode([job_text])
            semantic_score = cosine_similarity(r_emb, j_emb)[0][0] * 100

            # Skills
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            matched = list(set(resume_skills) & set(job_skills))
            missing = list(set(job_skills) - set(resume_skills))

            skill_score = (len(matched) / max(len(job_skills),1)) * 100
            keyword_score = (len(matched) / max(len(resume_skills),1)) * 100
            quality_score = min(len(resume_text.split()) / 500, 1) * 100

            ats_score = round(
                0.4*skill_score +
                0.3*semantic_score +
                0.2*keyword_score +
                0.1*quality_score,
                2
            )

        # ---------------- RESULTS ----------------
        st.success("Analysis Complete")

        st.markdown(f"""
        <div style="padding:20px;border-radius:15px;border:1px solid #2a2a2a;text-align:center;">
        <h2>📊 ATS Score</h2>
        <h1 style="color:#00ffcc;">{ats_score}%</h1>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- ATS BREAKDOWN ----------------
        st.subheader("📌 ATS Score Breakdown")
        st.progress(int(skill_score))
        st.write(f"Skill Match: {round(skill_score,2)}%")

        st.progress(int(semantic_score))
        st.write(f"Semantic Similarity: {round(semantic_score,2)}%")

        st.progress(int(keyword_score))
        st.write(f"Keyword Coverage: {round(keyword_score,2)}%")

        st.progress(int(quality_score))
        st.write(f"Resume Quality: {round(quality_score,2)}%")

        # ---------------- SKILLS ----------------
        colA, colB = st.columns(2)
        with colA:
            st.subheader("✅ Matching Skills")
            for s in matched: st.success(s)

        with colB:
            st.subheader("❌ Missing Skills")
            for s in missing: st.error(s)

        # ---------------- JOB RECOMMENDATION ----------------
        st.subheader("💼 Job Recommendations")

        job_scores = {}
        for role, skills in JOB_ROLES.items():
            score = len(set(skills) & set(resume_skills)) / len(skills)
            job_scores[role] = round(score * 100, 2)

        for role, score in sorted(job_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            st.info(f"**{role}** — {score}% match")

        # ---------------- AGENTIC AI ----------------
        st.subheader("🤖 AI Career Advice")

        if ats_score >= 80:
            st.success("Strong ATS profile. Focus on interviews & system design.")
        elif ats_score >= 60:
            st.warning(f"Improve these skills to boost ATS score: {', '.join(missing[:5])}")
        else:
            st.error("Low ATS score. Tailor resume and build role-specific projects.")

# ---------------- FOOTER ----------------
st.markdown("<hr><p style='text-align:center;'>AI-Powered Hiring Intelligence</p>", unsafe_allow_html=True)
