from loaders.resume_jobs import ResumeLoaderChroma, ResumeLoaderFAISS
from pathlib import Path
import llms
import rags
import numpy as np

tested_models = ["deepseek-ai/DeepSeek-R1",
                 #  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 #  "meta-llama/Meta-Llama-3-8B-Instruct",
                 "meta-llama/Llama-3.2-1B",
                 "meta-llama/Llama-3.2-1B-Instruct",
                 "meta-llama/Llama-3.1-8B",
                 #  "meta-llama/Llama-3.1-8B-Instruct",
                 "meta-llama/Llama-3.2-3B-Instruct",
                 #  "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                 "databricks/dolly-v2-3b",
                 #  "tiiuae/falcon-7b",
                 #  "tiiuae/falcon-7b-Instruct",
                 ]


if __name__ == "__main__":
    # Job => CV pool => Bring me 5 candidates score them. If not, Open
    # New CV => Analyze => score.
    #
    job_description_path = "embeddings/embeddingsBase/chroma/train/job_description_text"
    resume_text_path = "embeddings/embeddingsBase/chroma/train/resume_text"

    job_db = ResumeLoaderChroma(keep_in_memory=False)
    job_db.load_indexes(job_description_path)
    resume_text_db = ResumeLoaderChroma(keep_in_memory=False)
    resume_text_db.load_indexes(resume_text_path)
    # start pipeline

    # llm = llms.build_llm(model="meta-llama/Llama-2-7b")
    llm = llms.LLMScorer(model="meta-llama/Llama-3.2-1B-Instruct")
    jobs = {}
    sample = rags.get_sample(job_db, 0)
    cv_assistant = rags.RAGSystem(db=resume_text_db,
                                  vectorstore=resume_text_db.vectors,
                                  search_type="similarity",
                                  k=25)
    docs = cv_assistant.get_relevant_cvs(sample["sample"]['job_description_text'],
                                         llm=llm.analyzer,)
    print(f"We found {len(docs)} possible candidates")
    scores = {}
    for idx, doc in docs.items():
        score = cv_assistant.evaluate_cv_job_pairwise(resume=doc,
                                                      job_description=sample["sample"]['job_description_text'],
                                                      llm=llm.judge,
                                                      )
        scores[idx] = score
    print(scores)
