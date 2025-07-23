# Resume Analyser

Resume Analyzer is an intelligent evaluation tool that leverages Retrieval-Augmented Generation (RAG) to assess candidate profiles with context-aware precision. Unlike basic keyword matching, this system understands the skills, experience, and qualifications behind a resume to produce quantitative scores and qualitative explanations, empowering recruiters, HR tech platforms, and hiring teams to make informed decisions (assigning each a quantitative compatibility score and a qualitative explanation):

1. Candidate Matching: Given a new job description and a resume database, retrieve the top K most relevant candidates (by ID) based on semantic fit.
2. Cold Resume Evaluation: If no existing candidates meet predefined thresholds, evaluate a newly submitted resume against the job role and return a compatibility score.
3. Reverse Matching: For any incoming resume, identify the best-matching job opening from a job description database and retrieve the top K matches.

## Key Features

Langchain: Wraps Hugging Face models and simplifies prompt handling, vector store access, and LLM orchestration.

Retrieval Augmented Generation (RAG): ombines a high-performance vector retriever (e.g., FAISS or ChromaDB) with an LLM backbone (via LangChain’s HuggingFacePipeline). The system first retrieves the most semantically relevant documents (Resume or job descriptions) with their IDs, then feeds them into the LLM to generate grounded compatibility scores and concise, explainable summaries.

Databases libraries: FAISS & ChromaDB for comparison in this project. The next project would have: Pinecone, Weaviate, Qdrant, Milvus :) Reference: [(Vector DB comparison)](https://medium.com/tech-ai-made-easy/vector-database-comparison-pinecone-vs-weaviate-vs-qdrant-vs-faiss-vs-milvus-vs-chroma-2025-15bf152f891d). Note that Metadata and other infos in Chroma were not included for simplicity but will be added in the next project.

Specifically, Added:

- [X] Database loader
- [X] Indexing via Chroma & FAISS vector databases.
- [X] Load LLMs pipeline
- [X] Simple RAG pipeline implementation
- [X] Adding Analyzer and Judge LLMs (summary evaluation and score)
- [X] Task 1: Given a job description find a "good match" Resume in a database
- [X] Task 2: Set a score to a new Resume entry vs a job description
- [ ] Task 3: Get system metrics given a database.
- [ ] Visual Interface

## Install

```bash
cd ResumeAnalyzer
python3.10 -m venv prj2
source prj2/bin/activate
pip install --upgrade pip
pip install uv
uv init . 
uv sync --active
```

Important note for FAISS (GPU): [issue with numpy](https://github.com/facebookresearch/faiss/issues/3526)

## Run

```bash
huggingface-cli login
```

### Precompute embeddings

To compute embeddings from the database `ahmedheakl/resume-atlas` and `Lakshmi12/Resume_Dataset` or job description dataset, run variation of the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python src/loaders/resume_classifier.py --model "intfloat/e5-large-v2" --output_path "embeddings/embeddingsE5Large" --split "train"

CUDA_VISIBLE_DEVICES=1 python src/loaders/resume_classifier.py --model "sentence-transformers/all-MiniLM-L6-v2" --output_path "embeddings/embeddingsMini" --split "train"

CUDA_VISIBLE_DEVICES=0 python src/loaders/resume_jobs.py --model "intfloat/e5-large-v2" --output_path "embeddings/resume_jobs/embeddingsE5Large" --split "train"
```

Some classes that are contained in the dataset (cleaned) labels including the following:

```bash
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARCHITECTURE, ARTS, AUTOMOBILE, AVIATION, BANKING, BLOCKCHAIN, Business Process Outsourcing, BUILDING AND CONSTRUCTION, BUSINESS ANALYST, BUSINESS DEVELOPMENT, CHEF, CIVIL ENGINEER, CONSTRUCTION, CONSULTANT, DATA SCIENCE, DATABASE, DESIGNER, DESIGNING, DEVOPS, DIGITAL MEDIA, DOTNET DEVELOPER, EDUCATION, ELECTRICAL ENGINEERING, ENGINEERING, ETL DEVELOPER, FINANCE, FITNESS, FOOD AND BEVERAGES, HEALTH AND FITNESS, HEALTHCARE, HUMAN RESOURCES, INFORMATION TECHNOLOGY, JAVA DEVELOPER, MANAGEMENT, MECHANICAL ENGINEER, NETWORK SECURITY ENGINEER, OPERATIONS MANAGER, Project Management Office, PUBLIC RELATIONS, PYTHON DEVELOPER, REACT DEVELOPER, SALES, SAP DEVELOPER, SQL DEVELOPER, TEACHER, TESTING, WEB DESIGNING
```