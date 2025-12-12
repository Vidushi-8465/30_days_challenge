import os                      #It lets us look inside the folders on the computer
import re                      # Regex for finding the experience
import json                    # To import the Json file that includes all the skills required for a job 
import spacy                   # It is called as language brain, It reads English sentences
import pandas as pd            # Used to handle the data by making tables 
import numpy as np             # Used to handle the data by doing the maths properly 
from spacy.matcher import PhraseMatcher  #Used to find specific words inside the resume
from sklearn.preprocessing import MinMaxScaler  #To find the numbers between 0 and  1 or used for normalization
from typing import List, Dict, Any              #

# Load NLP model i.e english lang model
#Spacy can : Read sentences, Split Words, Understand Text
nlp = spacy.load("en_core_web_sm")

# we create a class where we keep all the tools ready
class ResumeScorerSystem:

    #Setup
    def __init__(self, skill_taxonomy_path: str, role_required_skills: List[str] = None):
        # Load skill taxonomy (tiers + aliases)
        with open(skill_taxonomy_path, "r", encoding="utf-8") as f:
            self.skill_taxonomy = json.load(f)     # we are now loading the json file

        # Build matcher for skills
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  #It scans for the words that are present in the json file.
        
        #Add all the skills to the Matcher
        patterns = []
        for skill, meta in self.skill_taxonomy.items():
            for alias in meta.get("aliases", []) + [skill]:
                patterns.append(nlp.make_doc(alias))
        self.matcher.add("SKILLS", patterns)

        # Save skills required for a specific job role
        self.role_required_skills = [s.lower() for s in (role_required_skills or [])]

        # Distribute the weights for the final calculation
        self.WEIGHTS = {
            "skill_match": 0.4,
            "experience": 0.2,
            "project_complexity": 0.25,
            "diversity": 0.15             # This shows that Skills matter the most , followed by experience and others
        }

        # Words indicating complex projects i.e if these words are found in the resume , then that shows that the proj done is harder 
        # Therefore more Points
        self.COMPLEXITY_KEYWORDS = [
            "deep", "complex", "distributed", "scalable", "microservice",
            "nlp", "neural", "transformer", "kubernetes", "ci/cd"
        ]  #These are few of the words, can be changed later as per the requirements.

    # PARSING SECTION , where we read the text i. resumes
    # we give the resue to spacy which reads it like a story
    def parse_resume(self, text: str) -> Dict[str, Any]:
        doc = nlp(text)

        # Extract  i.e while reading/parsing the text if we find the required skill we extract and keep it in (skills)
        matches = self.matcher(doc)
        skills = set()
        for _, start, end in matches:
            skills.add(doc[start:end].text.lower())

        # Extract years of experience i.e we look for (__yrs)
        exp_years = self.extract_experience(text)

        # Extract project descriptions i.e read all projects from the resume
        projects = self.extract_projects(text)

        # We return all the things like skills, exp years, projects and the resume.
        return {   
            "raw_text": text,
            "skills": list(skills),
            "experience_years": exp_years,
            "projects": projects
        }
    # We look for pattern like : __years,__year,__yr,__yrs using regex
    def extract_experience(self, text: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(years|year|yrs|yr)", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        # or even sometimes the resume says: experience/s: ___
        m2 = re.search(r"experience[:\s]+(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m2:
            return float(m2.group(1))
        # if nthg is found we return 0
        return 0.0
    # we look for the project section 
    def extract_projects(self, text: str) -> List[Dict[str, Any]]:

        projects = []
        if "projects:" in text.lower():
           # to split the text into lines
            _, psec = text.split("Projects:", 1)
           # each line is treated as a separate line
            lines = [l.strip() for l in psec.splitlines() if l.strip()]

            for line in lines:
                # we will now extract the duration in the brackets i.e (3 months / 6 months) etc.
                dur_m = re.search(r"\((\d+(?:\.\d+)?)\s*(months|month|years|year)\)", line, flags=re.IGNORECASE)
                duration_months = 0
                if dur_m:
                    val = float(dur_m.group(1))
                    unit = dur_m.group(2).lower()
                     
                    # 1 year= 12 months
                    if "year" in unit:
                        duration_months = val * 12
                    else:
                        duration_months = val
                # we then store each projet
                projects.append({
                    "description": line,
                    "duration_months": duration_months
                })
        # to return the list.
        return projects

    # SCORING SECTION
    # we now take the parsed resume and score it
    def score_resume(self, parsed_resume: Dict[str, Any]) -> Dict[str, float]:
        skills = [s.lower() for s in parsed_resume["skills"]]
        exp_years = parsed_resume["experience_years"]
        projects = parsed_resume["projects"]

        # 1. Skill Match
        # for eg: if the job requires 10 skills and i have 7, so the score for skills would be : 7/10 = 0.7
        if self.role_required_skills:
            matched = sum(1 for r in self.role_required_skills if r in skills)
            skill_match = matched / max(1, len(self.role_required_skills))
        # else we will use skill tier i.e beginner= 1, medium=2, advanced=3
        else:
            tier_scores = []
            for s in skills:
                tier_scores.append(self.skill_taxonomy.get(s, {}).get("tier", 1))
            skill_match = sum(tier_scores) / (3 * max(1, len(tier_scores))) if tier_scores else 0

        # 2️. Experience Score (smooth)
        # we use the function tanh which smoothes out the score so that only 0- 5 matters the most
        exp_score = np.tanh(exp_years / 5)

        # 3️. Project Complexity
        complexity_score = 0  # it starts at 0

        # look for complex keywords
        for p in projects:
            desc = p["description"].lower()
            dur = p["duration_months"]
            keyword_hits = sum(1 for kw in self.COMPLEXITY_KEYWORDS if kw in desc)

        # add keyword + duration score. i.e harder proj = more score
            complexity_score += (keyword_hits * 0.5) + (dur / 12 * 0.1)
        
        # we then normalize the score and make it between 0 and 1
        if projects:
            complexity_score /= (3 * len(projects))
            complexity_score = min(1.0, complexity_score)

        # 4️. Diversity i.e if the resume has a mixture of beg+med +adv skills = more diversity
        unique_tiers = set()
        for s in skills:
            unique_tiers.add(self.skill_taxonomy.get(s, {}).get("tier", 1))
        diversity_score = min(1.0, len(unique_tiers) / 3)

        # 5️. Final Weighted Score
        # that we mix all the scores together.
        final_score = (
            self.WEIGHTS["skill_match"] * skill_match +
            self.WEIGHTS["experience"] * exp_score +
            self.WEIGHTS["project_complexity"] * complexity_score +
            self.WEIGHTS["diversity"] * diversity_score
        )

        return {
            "skill_match": skill_match,
            "experience": exp_score,
            "project_complexity": complexity_score,
            "diversity": diversity_score,
            "raw_score": final_score
        }

    # END-TO-END PIPELINE
    
    # 1. we will score each resume in the folder
    def process_folder(self, folder_path: str, output_csv: str):
        records = []
        
        # 2. Open each resume and read it
        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                    text = f.read()
                
                # 3. Parse + Score 
                parsed = self.parse_resume(text)
                scored = self.score_resume(parsed)
                
                # 4. Save
                scored["filename"] = fname
                records.append(scored)
        
        #5. Create a dataframe i.re table using pandas
        df = pd.DataFrame(records)

        # Normalize raw scores using MinMaxScaler
        scaler = MinMaxScaler()
        df["normalized_score"] = scaler.fit_transform(df[["raw_score"]])

        df.sort_values("normalized_score", ascending=False, inplace=True)
        df.to_csv(output_csv, index=False)

        print("\n✔ Resume scoring completed!")
        print(f"✔ Output saved to: {output_csv}")

def main():
    scorer = ResumeScorerSystem(
        skill_taxonomy_path="skill_taxonomy.json",
        role_required_skills=["python", "machine learning", "sql"]
    )

    # Read the resume
    with open("resume1.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Parse → Score
    parsed = scorer.parse_resume(text)
    scored = scorer.score_resume(parsed)

    # Print results
    print("\n----- RESULTS -----")
    print("Skills Found:", parsed["skills"])
    print("Experience Years:", parsed["experience_years"])
    print("Project Count:", len(parsed["projects"]))
    print("\nRaw Score:", scored["raw_score"])
    print("--------------------")

    return scored["raw_score"]

def compare_two_resumes():
    scorer = ResumeScorerSystem(
        skill_taxonomy_path="skill_taxonomy.json",
        role_required_skills=["python", "machine learning", "sql"]
    )

    files = ["resume1.txt", "resume2.txt"]

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        parsed = scorer.parse_resume(text)
        scored = scorer.score_resume(parsed)

        print(f"\n----- RESULTS FOR {file} -----")
        print("Skills:", parsed["skills"])
        print("Experience:", parsed["experience_years"])
        print("Projects:", len(parsed["projects"]))
        print("Raw Score:", scored["raw_score"])

    print("\nComparison Complete!")


# Run program
# if __name__ == "__main__":
   # main()

if __name__ == "__main__":
    compare_two_resumes()
    
