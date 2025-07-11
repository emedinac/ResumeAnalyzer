# Resume Analyser

Resume Analyser is a tool that uses Retrieval-Augmented Generation (RAG) to deeply understand candidate profiles. Instead of just matching keywords, it connects skills, experience, and qualifications to provide clear, explainable insights for hiring decisions. Perfect for recruiters, HR tech platforms, and anyone looking to evaluate talent more effectively.

## Key Features

Langchain: used to wraps Huggingface models and has everything inside it :D. So I can easily use with prompts.

Retrieval Augmented Generation (RAG):

Databases libraries: FAISS (fastest & ready-production tool) & ChromaDB (standard in langchain and quick protoypes) for comparison in this project. The next project would have: Pinecone, Weaviate, Qdrant, Milvus :) Reference: [(Vector DB comparison)](https://medium.com/tech-ai-made-easy/vector-database-comparison-pinecone-vs-weaviate-vs-qdrant-vs-faiss-vs-milvus-vs-chroma-2025-15bf152f891d). Note that Metadata and other infos in Chroma were not included for simplicity but will be added in the next project.

Specifically, Added:

- [X] Database loader
- [X] Indexing via Chroma & FAISS vector databases.
- [X] Load LLMs pipeline
- [ ] RAG implementation
- [ ] 

Important note for FAISS (GPU): [issue with numpy](https://github.com/facebookresearch/faiss/issues/3526)

## Install

```bash
python3.10 -m venv prj2
source prj2/bin/activate
cd ResumeAnalyzer
pip install --upgrade pip
pip install uv
uv init . 
uv sync --active
```

## Run

```bash
huggingface-cli login
```
