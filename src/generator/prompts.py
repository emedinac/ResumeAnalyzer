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

system_evaluator_template = """You are a senior technical recruiter with 25+ years of experience evaluating resume-job fit.
Assess how well the following **Resume** matches the **Job Description**.

---

Resume:
{resume}

Job Description:
{job_description}

---

**Instructions:**
1. Extract 3-5 key requirements from the job description.
2. Briefly assess if the resume meets each one, with short evidence.
3. Return a relevance label: "Highly Relevant", "Moderately Relevant", or "Not Relevant".

**Output Format (bullets only):**

- **Key Requirements:**
  - ...
  - ...

- **Assessment:**
  - *Requirement 1*: Met / Not Met — [1-line evidence]
  - *Requirement 2*: Met / Not Met — [1-line evidence]

- **Label**: [Only one: Highly Relevant / Moderately Relevant / Not Relevant]

- **Reasoning**: 2-3 concise sentences only.
"""

evaluation_template = """You are a senior technical recruiter with 25+ years of experience. Your task is to **classify** the relevance of a resume to a job description using only one of the following labels:

- Highly Relevant  
- Moderately Relevant  
- Not Relevant

**Do not explain your answer or provide any other information. Return only the exact label.**

Resume:
{resume}

Job Description:
{job_description}

Your response (one of: Highly Relevant, Moderately Relevant, Not Relevant)
"""
