#  day 20 : AI Job Matching & ATS Scoring System

##  Project Overview

This project is an **AI-powered Job Matching System** that analyzes a candidate’s resume and compares it with a job description to determine suitability.
It goes beyond keyword matching by using **semantic embeddings**, **ATS-style scoring**, **skill gap analysis**, and **career recommendations**.

The system helps candidates understand:

* How well their resume matches a job role
* Which skills are missing
* Their ATS score breakdown
* Recommended job roles based on their profile

---

##  Key Features

*  PDF resume upload and parsing
*  Semantic similarity using embeddings
*  ATS score with detailed breakdown
*  Matching &  missing skills detection
*  Job role recommendations
*  Agentic AI-based career advice

---

## 🛠 Technologies Used

* **Python**
* **Streamlit** – UI & web interface
* **Sentence Transformers** – text embeddings
* **Scikit-learn** – cosine similarity
* **PyPDF2** – resume PDF parsing

---

##  Algorithms & Concepts Used

### 1. Semantic Similarity (Embeddings)

* **Model:** `all-MiniLM-L6-v2`
* Converts resume and job description into vector embeddings
* Uses **Cosine Similarity** to measure meaning-based match

### 2. ATS Scoring Algorithm

Weighted scoring based on:

* Skill Match (40%)
* Semantic Similarity (30%)
* Keyword Coverage (20%)
* Resume Quality (10%)

This mimics real-world **Applicant Tracking Systems (ATS)**.

### 3. Skill Matching Logic

* Extracts skills using keyword matching
* Compares resume skills with job-required skills
* Identifies matching and missing skills

### 4. Job Recommendation Algorithm

* Content-based recommendation
* Matches resume skills with predefined job role skill sets
* Ranks top suitable roles

---

##  Important Functions Used

* `extract_text_from_pdf()` – Extracts text from resume PDF
* `clean_text()` – Preprocesses text for better matching
* `extract_skills()` – Identifies skills from resume and job description
* `cosine_similarity()` – Computes semantic similarity score

---

##  Use Cases

* Resume screening systems
* Career guidance platforms
* ATS optimization tools
* AI-based hiring assistants

---

##  Future Enhancements

* LLM API integration for resume improvement
* Multi-job comparison
* Resume rewrite suggestions
* Recruiter dashboard
* LangChain-based multi-agent system

---

 *This project demonstrates practical use of AI, NLP, and system design concepts in real-world hiring workflows.*


