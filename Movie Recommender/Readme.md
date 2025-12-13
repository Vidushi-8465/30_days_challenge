## Day 4 Proj 2: Movie Recommendation System

A simple **content-based movie recommender** using TF-IDF and cosine similarity.
This project recommends movies that are similar to a given movie using the textual metadata (genres + overview). 
It is a content-based recommender: for a chosen movie it finds the most similar movies based on text.

---
## 🔹 Features
* Extract skills, experience, and projects from resumes.
* Score resumes based on:

  * Skill match
  * Experience
  * Project complexity
  * Skill diversity
* Normalize scores for fair comparisons.
* Compare two or more resumes at once.
* Process all resumes in a folder and output results in a CSV.
---

## 🛠 Tech Stack & Libraries
* **Python 3.x**
* **SpaCy** → For NLP (reading, tokenizing, and understanding resumes)
* **Pandas** → Data handling and creating tables
* **NumPy** → Math operations
* **scikit-learn** → MinMaxScaler for score normalization
---

## Folder Structure

```
Resume_Scorer/
│
├─ resume_scorer.py        # Main Python script (all code in single file)
├─ skill_taxonomy.json     # JSON file with skills, aliases, and tiers
├─ resume.txt              # Sample resume text file
├─ resume1.txt             # Second sample resume (optional)
├─ resume2.txt             # Third sample resume (optional)
└─ output.csv              # Output file for batch processing resumes
```

---

## 🔹 Basic Flow of the Program

1. **Load NLP Model:**
   SpaCy English model (`en_core_web_sm`) is loaded to read and parse resume text.

2. **Load Skill Taxonomy:**
   A JSON file containing all relevant skills, aliases, and skill tiers is loaded to identify skills in resumes.

3. **Parse Resume:**

   * Extract skills using `PhraseMatcher`.
   * Extract experience in years using regex.
   * Extract project details and duration.

4. **Score Resume:**
   Each resume is scored based on:

   * **Skill Match:** Matches role-required skills or uses tier-based scoring.
   * **Experience Score:** Smoothed using `tanh` to limit extreme values.
   * **Project Complexity:** Based on keywords (like NLP, Kubernetes) and project duration.
   * **Diversity Score:** Measures how many different skill tiers are present.
   * **Final Weighted Score:** Weighted combination of all factors.

5. **Normalize Scores:**
   MinMaxScaler ensures all scores are between 0 and 1.

6. **Output:**
   Scores are printed and/or saved in a CSV file for batch processing.

---

## 🔹 Important Functions

| Function                                  | Purpose            | Explanation                                                                                  |
| ----------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------- |
| `parse_resume(text)`                      | Parse a resume     | Reads text, extracts skills, experience, and projects.                                       |
| `extract_experience(text)`                | Extract experience | Uses regex to find years of experience mentioned in resume.                                  |
| `extract_projects(text)`                  | Extract projects   | Extracts project descriptions and durations from the resume.                                 |
| `score_resume(parsed_resume)`             | Score resume       | Calculates skill match, experience, project complexity, diversity, and final weighted score. |
| `process_folder(folder_path, output_csv)` | Batch processing   | Scores all resumes in a folder and saves output CSV.                                         |

---

## 🔹 Algorithms & Techniques Used

* **Keyword Matching:** Using SpaCy `PhraseMatcher` to extract skills.
* **Regex Parsing:** To find years of experience and project durations.
* **Weighted Scoring:** Combines multiple factors to produce a single score.
* **Score Normalization:** `MinMaxScaler` ensures unbiased comparison across resumes.
* **Complexity Detection:** Checks project descriptions for keywords like `NLP`, `Kubernetes`, `Distributed`, etc.
* **Diversity Metric:** Rewards candidates with a mix of skill tiers.

---

## 🔹 Comparing Two Resumes

* You can compare two resumes using the function `compare_two_resumes()`
* Reads `resume1.txt` and `resume2.txt` (or any two text files)
* Prints out:

  * Skills found
  * Years of experience
  * Number of projects
  * Raw score

**Example Output:**

```
----- RESULTS FOR resume1.txt -----
Skills: ['python', 'sql']
Experience: 2
Projects: 3
Raw Score: 0.72

----- RESULTS FOR resume2.txt -----
Skills: ['python', 'machine learning', 'docker']
Experience: 4
Projects: 4
Raw Score: 0.85
```

The **higher raw score** indicates the better candidate.

---

## 🔹 Batch Processing All Resumes

* Place all `.txt` resumes inside a folder (e.g., `resumes/`)
* Call:

```python
scorer.process_folder("resumes", "output.csv")
```

* **Output:** `output.csv` containing normalized scores and rankings.

---

## 🔹 Sample JSON (skill_taxonomy.json)

```json
{
  "python": {"tier": 3, "aliases": ["py"]},
  "sql": {"tier": 2, "aliases": ["structured query language"]},
  "machine learning": {"tier": 3, "aliases": ["ml"]},
  "docker": {"tier": 2, "aliases": []}
}
```

---

## 🔹 How to Run

1. Install dependencies:

```bash
pip install spacy pandas numpy scikit-learn
python -m spacy download en_core_web_sm
```

2. Run single resume:

```bash
py resume_scorer.py
```

3. Compare two resumes:

```python
if __name__ == "__main__":
    compare_two_resumes()
```

4. Process folder:

```python
scorer.process_folder("resumes", "output.csv")
```

---
## 🔹 Future Improvements
* Add **PDF resume parsing** using `PyPDF2` or `pdfplumber`.
* Build a **GUI/Streamlit web app** for uploading resumes.
* Extend **skill taxonomy** for multiple job roles.
* Add **AI-based semantic skill matching** instead of exact keyword matching.
---