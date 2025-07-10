job_description = None  # to avoid warning in VSCode :D

template1 = f"""Given this job description:
\"{job_description}\"
Evaluate the following extracted CV based on:
- Skill match 
- Relevant years of experience
- Key achievements
- Educational level
Return a score between 0-100 and a brief summary explaining the score for this applicant:
"""

template2 = f"""Given this job description:
\"{job_description}\"
Return a score between 0-100 and a brief summary explaining the score for this applicant:
"""

template3 = f"""Job description:
\"{job_description}\"
Return a score between 0-100 for this applicant:
"""

template4 = f"""Job description:
\"{job_description}\"
Return a score (0-100) for this applicant:
"""


template5 = f"""Job description:
\"{job_description}\"
Return only a score (0-100) for this applicant:
"""
