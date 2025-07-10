from loader import ResumeLoaderChroma, ResumeLoaderFAISS
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
    job_description_path = "embeddings/embeddingsBase/chroma/train/job_description_text"
    resume_text_path = "embeddings/embeddingsBase/chroma/train/resume_text"

    job_db = ResumeLoaderChroma()
    job_db.load_indexes(job_description_path)
    resume_text_db = ResumeLoaderChroma()
    resume_text_db.load_indexes(resume_text_path)
    # llm = llms.build_llm(model="meta-llama/Llama-2-7b")
    llm = llms.build_llm(model="meta-llama/Llama-3.2-1B-Instruct")
    rag = rags.build_rag(vectorstore=resume_text_db.vectors,
                         llm=llm,
                         search_type="similarity",
                         k=5
                         )
    jobs = job_db.vectors._collection.get(include=[])
    jobs_ids = jobs["ids"]
    jobid = jobs_ids[np.random.randint(len(jobs_ids))]
    job_description = job_db.vectors.get(ids=[jobid])
    score = rags.evaluate_cv_with_offer(rag,
                                        job_description=job_description["documents"][0],
                                        prompt_template=rags.prompts.template1
                                        )
    print(score)
