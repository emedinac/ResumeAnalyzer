# context is the job description
# question is the CV Resume

evaluation_template = """You are an evaluation assistant tasked with reviewing the output from a CV analysis model.

Here is the candidate's CV:
{context}

Target job role:
{role}

Here is the model's evaluation:
SCORE: {score}
SUMMARY: {summary}
POTENTIAL_FIT_AREAS: {fit_areas}

Your task is to assess the accuracy and validity of the model's evaluation.

Evaluate the following:

1. Is the SCORE appropriate based on the content of the CV and relevance to the target role?
2. Does the SUMMARY accurately reflect the candidate's experience, and is it clearly supported by the CV?
3. Are the suggested POTENTIAL_FIT_AREAS logical and supported by evidence in the CV? The possible areas to match are strictly limited to: ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARCHITECTURE, ARTS, AUTOMOBILE, AVIATION, BANKING, BLOCKCHAIN, Business Process Outsourcing, BUILDING AND CONSTRUCTION, BUSINESS ANALYST, BUSINESS DEVELOPMENT, CHEF, CIVIL ENGINEER, CONSTRUCTION, CONSULTANT, DATA SCIENCE, DATABASE, DESIGNER, DESIGNING, DEVOPS, DIGITAL MEDIA, DOTNET DEVELOPER, EDUCATION, ELECTRICAL ENGINEERING, ENGINEERING, ETL DEVELOPER, FINANCE, FITNESS, FOOD AND BEVERAGES, HEALTH AND FITNESS, HEALTHCARE, HUMAN RESOURCES, INFORMATION TECHNOLOGY, JAVA DEVELOPER, MANAGEMENT, MECHANICAL ENGINEER, NETWORK SECURITY ENGINEER, OPERATIONS MANAGER, Project Management Office, PUBLIC RELATIONS, PYTHON DEVELOPER, REACT DEVELOPER, SALES, SAP DEVELOPER, SQL DEVELOPER, TEACHER, TESTING, WEB DESIGNING

Return your evaluation in this exact format:

- VALID_SCORE: Yes or No
- VALID_SUMMARY: Yes or No
- VALID_POTENTIAL_FIT_AREAS: Yes or No
- RECOMMENDED_CLASS: <Select the **single most appropriate category** from POTENTIAL_FIT_AREAS that aligns **best with the target job role**>
- ERRORS_OR_INCONSISTENCIES: <Brief explanation of any issue found, or "None">
- RECOMMENDED_SCORE (optional): <Only if the score is incorrect>

Do not guess outside the POTENTIAL_FIT_AREAS list. Do not invent categories. Be strict.

ANSWER:
"""


system_template = """You are a thorough and structured assistant for CV evaluations. 
Answer the user's query based on the "query" below.
If you cannot answer the question using the provided information answer with "I don't know. I need more information".
"""

extraction_template = """Given the following job title or role description:

{context}

Return a concise list of the most relevant keywords, tools, frameworks, or core skills typically associated with this role.

Follow these rules:
- Output a single, comma-separated list.
- Use only one word or a short phrase per item (max 3 words).
- No duplicate or overly similar items (e.g., only one version of that word).
- Limit to the top **10 distinct and essential** items only.
- Avoid overly generic or vague terms (e.g., "skills", "tools", "tech").
- Do not categorize or explain anything.
- Only output the list. Nothing else.

ANSWER:
"""

cv_question1 = """EVALUATION TASK: You are evaluating a candidate's CV for potential fit into predefined job categories.

ROLE TITLE: {role}

--------

CANDIDATE CV: {context}

--------

YOUR TASK:

- Analyze the candidate's experience, skills, and qualifications from the CV.
- Choose the ONE BEST-FIT category from the following list that most accurately reflects the candidate's primary field or expertise:

ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARCHITECTURE, ARTS, AUTOMOBILE, AVIATION, BANKING, BLOCKCHAIN, Business Process Outsourcing, BUILDING AND CONSTRUCTION, BUSINESS ANALYST, BUSINESS DEVELOPMENT, CHEF, CIVIL ENGINEER, CONSTRUCTION, CONSULTANT, DATA SCIENCE, DATABASE, DESIGNER, DESIGNING, DEVOPS, DIGITAL MEDIA, DOTNET DEVELOPER, EDUCATION, ELECTRICAL ENGINEERING, ENGINEERING, ETL DEVELOPER, FINANCE, FITNESS, FOOD AND BEVERAGES, HEALTH AND FITNESS, HEALTHCARE, HUMAN RESOURCES, INFORMATION TECHNOLOGY, JAVA DEVELOPER, MANAGEMENT, MECHANICAL ENGINEER, NETWORK SECURITY ENGINEER, OPERATIONS MANAGER, Project Management Office, PUBLIC RELATIONS, PYTHON DEVELOPER, REACT DEVELOPER, SALES, SAP DEVELOPER, SQL DEVELOPER, TEACHER, TESTING, WEB DESIGNING

- You MUST select only one (1) category from the above list that is most applicable.
- If no clear match exists, return: []

SCORE:
- Provide a number from 0 to 100 indicating how well the candidate fits the specific ROLE TITLE.
- Use a strict and conservative scoring approach:
    - 90-100: Candidate is an **exceptional match** with strong evidence across multiple relevant experiences.
    - 70-89: Candidate is a **good match** but may lack full alignment or depth.
    - 50-69: Candidate is a **partial match**, with limited or tangential relevance.
    - Below 50: Candidate is **not a match** or lacks sufficient information for evaluation.

Then, provide:
- SCORE: A number from 0 to 100  
- POTENTIAL_FIT_AREAS: A list of **up to 5** other relevant categories from the list (comma-separated), based on the candidate's secondary skills or experiences.
- SUMMARY: A concise, factual 2-3 sentence explanation of why the candidate fits (or does not fit) the role and category.

Output format (strictly follow this format):

SCORE: <number from 0 to 100>  
POTENTIAL_FIT_AREAS: <comma-separated list of up to 5 categories>  
SUMMARY: <2-3 factual sentences summarizing the candidate's fit>

Do not explain the task or provide any headings.

ANSWER:
"""
