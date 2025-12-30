def generate_prompt(skills, job_desc, difficulty, q_type):
    return f"""
You are a senior technical interviewer.

Candidate skills:
{skills}

Job description:
{job_desc}

Generate {difficulty} level {q_type} interview questions.

Requirements:
- Generate 5 questions
- Questions must be practical and role-specific
- Include 1 follow-up question for each
- Provide short answer hints
- Avoid generic textbook questions
"""
