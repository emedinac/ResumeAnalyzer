from loaders import resume_jobs, resume_classifier
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
    resume_text_path = "embeddings/resume_classifier/embeddingsBase/chroma/train/Resume"
    resume_text_db = resume_classifier.ResumeLoaderChroma(keep_in_memory=False,
                                                          model_name="sentence-transformers/all-mpnet-base-v2",
                                                          )
    resume_text_db.load_indexes(resume_text_path)
    # start pipeline
    sample = rags.get_sample(resume_text_db, 0)
    requirements = sample["sample"]["Category"]

    llm = llms.LLMScorer(model="meta-llama/Llama-3.2-1B-Instruct")
    jobs = {}

    cv_assistant = rags.RAGSystem(db=resume_text_db,
                                  vectorstore=resume_text_db.vectors,
                                  search_type="similarity_score_threshold",
                                  search_kwargs={'score_threshold': 0.65,
                                                 "k": 5},
                                  )

    # In case, we dont have to write them.
    # Compute them using an LLM to aggregate data to the query.
    if not ("," in requirements or len(requirements) > 20 or ";" in requirements):
        job_requirements = cv_assistant.get_relevant_requeriments(requirements,
                                                                  llm.analyzer)
    else:
        job_requirements = requirements
    candidates = cv_assistant.get_relevant_cvs(job_requirements, llm.analyzer)
    print(f"I found {len(candidates)} possible candidates\n")
    final_evaluation = {}
    for id_candidate, candidate in candidates.items():
        evaluation = cv_assistant.evaluate_cv_job_pairwise(candidate["resume"],
                                                           requirements,
                                                           candidate["evaluation"],
                                                           llm.judge
                                                           )
        if "INVALID" in evaluation:
            continue
        final_evaluation[id_candidate] = evaluation
    final_evaluation = rags.reorder_candidates_based_on_scores(
        final_evaluation)
    for id_candidate, candidate in final_evaluation.items():
        job_role = rags.get_sample(resume_text_db, int(
            id_candidate))["sample"]["Category"]
        print(f"db role label: {job_role}\n{candidate}\n\n")
    final_evaluation
