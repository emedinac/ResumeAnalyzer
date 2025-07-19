# GENERATION PROMPTS
gen_template_0 = """Design a job ad targeted at professionals with this profile.
Provide title, role overview, duties, required skills. Ensure unbiased phrasing.
"""
gen_template_1 = """Generate a job description tailored to the following candidate background.
Include responsibilities, qualifications, and experience requirements.
Use neutral, inclusive language. Do NOT infer gender, age, or ethnicity.
"""

gen_template_2 = """Create a realistic, fictional job description based only on this resume's skills and experience.
Avoid mentioning personal demographics or stereotypes.
"""

gen_template_3 = """You are a talent acquisition specialist.
Based on the resume below, write a job description including title, summary, responsibilities, and requirements.
Make it inclusive and unbiased.
"""

gen_template_4 = """Act as a hiring manager.
Create a role aligned with the candidate's skills and experience.
Include job title, key duties, and required qualifications without implying any demographic preference.
"""

gen_template_5 = """You are an HR partner designing a position suited to this candidate.
Produce title, overview, responsibilities, and skills needed—all in neutral, professional tone.
"""

gen_template_6 = """Using this resume, generate a concise job posting: title, mission statement, tasks, and qualifications.
Ensure language is inclusive and stereotype-free.
"""

gen_template_7 = """Act as a recruiter.
From this resume, build a vacancy post featuring title, job summary, duties, and required competencies—written neutrally and without bias.
"""

gen_template_8 = """You're structuring a public-sector job based on the candidate's background.
Provide descriptive title, purpose, responsibilities, and skill requirements—all in inclusive wording.
"""

gen_template_9 = """Create a position description that leverages the candidate's statistical and research skills.
Include overview, responsibilities, and must-have qualifications. Stay neutral and inclusive.
"""

# FORMAT PROMPTS
format_template_1 = """Include bullet points for requirements and responsibilities.
"""
format_template_2 = """Use numbered lists (not bullets) for qualifications.
"""
format_template_3 = """Use an optimistic and motivational tone that emphasizes mission and impact.
"""
format_template_4 = """Use a formal corporate tone.
"""
format_template_5 = """Frame it as a remote-friendly position.
"""

# NOISE PROMPTS

noise_template_0 = """End with a brief closing inviting applications ('Join us', etc.).
"""
noise_template_1 = """Structure it in 4 sections: Summary, Responsabilities, Required Skills and Experience, Nice-to-Have.
"""
noise_template_2 = """Frame the job as fully remote or hybrid, and note timezone flexibility.
"""
noise_template_3 = """Target the job description to mid-level professionals.
"""
noise_template_4 = """Target the job description to junior level professionals only.
"""
noise_template_5 = """Include expected years of experience as a numeric range.
"""
noise_template_6 = """Mention cross-functional collaboration in responsibilities.
"""
noise_template_7 = """Add one line on growth opportunities. Add one line including some salary range values or salary band (USD or EUR)
"""
noise_template_8 = """Highlight any leadership or mentorship aspects.
"""
noise_template_9 = """Target the job description to Principal or high senior level professionals only.
"""

system_evaluator_template = """You are a **senior technical recruiter and hiring analyst** with over 25 years of experience assessing resume-job matches. 
You will evaluate how well the given **Job Description** aligns with the provided **Resume**, and you must be extremely precise and transparent in your reasoning.

Resume:
{resume}

Job Description:
{job_description}

**Process:**
1. **Identify 3-5 key requirements** from the Job Description (skills, experience, qualifications).
2. **For each requirement**, assess whether the Resume demonstrates that competence. Provide evidence from the resume (e.g., “STATA programming for pollution analysis”).
3. **Assign a relevance label**:
   - **Highly Relevant** - Resume clearly meets most key requirements with direct evidence.
   - **Moderately Relevant** - Resume meets some requirements but lacks full coverage or explicit evidence.
   - **Not Relevant** - Resume meets few or none of the key requirements.

**Output Format:**
- A JSON object with the following fields:
  - '"requirements"': list of key Job Description requirements you extracted
  - '"assessment"': list of objects ' "requirement": ..., "met": true/false, "evidence": "..." '
  - '"label"': one of '"Highly Relevant"', '"Moderately Relevant"', '"Not Relevant"'
  - '"reasoning"': a 2-3 sentence justification synthesizing the individual assessments

"""
evaluation_template = """You are a **senior technical recruiter and hiring analyst** with over 25 years of experience assessing resume-job matches. 
You will evaluate how well the given **Job Description** aligns with the provided **Resume**, and you must be extremely precise and transparent in your reasoning.
Classify the relevance of the following resume to the given job description.

Resume:
{resume}

Job Description:
{job_description}

Relevance (choose one): Highly Relevant, Moderately Relevant, Not Relevant
"""
