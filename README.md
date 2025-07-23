# Resume Analyser

Resume Analyser is a tool that uses Retrieval-Augmented Generation (RAG) to deeply understand candidate profiles. Instead of just matching keywords, it connects skills, experience, and qualifications to provide clear, explainable insights for hiring decisions. Perfect for recruiters, HR tech platforms, and anyone looking to evaluate talent more effectively. This project solve the following situations (assigning each a quantitative compatibility score and a qualitative explanation):

1. Given a novel job description and a Resume document database, retrieve the top K most relevant candidates, retain their document identifiers.
2. If the initial candidate set fails to meet predefined fit thresholds, ingest a newly submitted Resume, evaluate it against the job requirements.
3. For any incoming Resume, query the database of open job postings to identify the single best‐matching and top K most relevant positions, retain the corresponding job-description identifiers.

## Key Features

Langchain: used to wraps Huggingface models and has everything inside it :D. So I can easily use with prompts.

Retrieval Augmented Generation (RAG): ombines a high-performance vector retriever (e.g., FAISS or ChromaDB) with an LLM backbone (via LangChain’s HuggingFacePipeline). The system first retrieves the most semantically relevant documents (Resume or job descriptions) with their IDs, then feeds them into the LLM to generate grounded compatibility scores and concise, explainable summaries.

Databases libraries: FAISS & ChromaDB for comparison in this project. The next project would have: Pinecone, Weaviate, Qdrant, Milvus :) Reference: [(Vector DB comparison)](https://medium.com/tech-ai-made-easy/vector-database-comparison-pinecone-vs-weaviate-vs-qdrant-vs-faiss-vs-milvus-vs-chroma-2025-15bf152f891d). Note that Metadata and other infos in Chroma were not included for simplicity but will be added in the next project.

Specifically, Added:

- [X] Database loader
- [X] Indexing via Chroma & FAISS vector databases.
- [X] Load LLMs pipeline
- [X] Simple RAG pipeline implementation
- [X] Adding Analyzer and Judge LLMs (summary evaluation and score)
- [X] Task 1: Given a job description find a "good match" Resume in a database
- [ ] Task 2: Set a score to a new Resume entry vs a job description
- [ ] Task 3: A new Resume entry with job description database
- [ ] Adding Metrics
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