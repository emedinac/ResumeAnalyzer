# context is the job description
# question is the CV Resume

evaluation_template = """
job description:
{job_description}

Previous compupation - resume with preliminar evaluation (score) sometimes include a short summary (summary):
{resume}

Based on the resume and job description, classify the fit level using one of:
1. "No Fit"
2. "Potential Fit"
3. "Good Fit"
don't include explanations in your answer, no summary, nothing except the fit class below.
Respond with only one of: "No Fit", "Potential Fit", or "Good Fit"

Answer: 
"""

system_template = """
You are a thorough and structured assistant for CV evaluations. 
Answer the user's query based on the "job description" below.
If you cannot answer the question using the provided information answer with "I don't know. I need more information".
"""

cv_question1 = """
Given this Job Description:
{context}

Given this Candidate CV:
{input}

Evaluate the following extracted CV based on:
- Skill match 
- Relevant years of experience
- Key achievements
- Educational level
Return a concrete value score (SCORE: ) between 0-100 and a brief summary (SUMMARY: ) explaining the score for this applicant.
Answer:
"""

cv_question2 = """
Given this Job Description:
{context}

Given this Candidate CV:
{input}

Return a concrete value score (SCORE: ) between 0-100 and a brief summary (SUMMARY: ) explaining the score for this applicant.
Answer:
"""

cv_question3 = """
Given this Job Description:
{context}

Given this Candidate CV:
{input}

Return a concrete value score (SCORE: ) between 0-100 for this applicant.
Answer:
"""

cv_question4 = """
Given this Job Description:
{context}

Given this Candidate CV:
{input}

Return a concrete value score (SCORE: ) (0-100) for this applicant.
Answer:
"""


cv_question5 = """
Given this Job Description:
{context}

Given this Candidate CV:
{input}

Return only a concrete value score (SCORE: ) (0-100) for this applicant. no more information just the number.
Answer:
"""
