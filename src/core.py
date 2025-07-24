from loaders import resume_jobs, resume_classifier
import llms
import rags
from langchain_community.document_loaders import PyPDFLoader, TextLoader

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


def load_file(path: str) -> str:
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
        documents = loader.load()
    elif path.endswith(".txt"):
        loader = TextLoader(path)
        documents = loader.load()
    else:
        raise ValueError(
            "Unsupported file format. Only .pdf and .txt are supported.")
    # Combine all page contents into a single string
    full_text = "\n".join(doc.page_content for doc in documents)
    return full_text


def load_db(path):
    if "base" in path.lower():
        model_name = "sentence-transformers/all-mpnet-base-v2"
    elif "mini" in path.lower():
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif "5base" in path.lower():
        model_name = "intfloat/e5-base-v2"
    elif "5large" in path.lower():
        model_name = "intfloat/e5-large-v2"
    db = resume_classifier.ResumeLoaderChroma(keep_in_memory=False,
                                              model_name=model_name,
                                              )
    db.load_indexes(path)
    return db


def get_candidates_given_job(requirements, db, model_name="meta-llama/Llama-3.2-1B-Instruct"):
    llm = llms.LLMScorer(model=model_name)
    cv_assistant = rags.RAGSystem(db=db,
                                  vectorstore=db.vectors,
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
    final_candidates = {}
    for id_candidate, candidate in candidates.items():
        evaluation = cv_assistant.evaluate_cv_job_pairwise(candidate["resume"],
                                                           requirements,
                                                           candidate["evaluation"],
                                                           llm.judge
                                                           )
        if "INVALID" in evaluation:
            continue
        # evaluation = cv_assistant.filter_answer(evaluation)
        final_candidates[id_candidate] = evaluation
    final_candidates = rags.reorder_candidates_based_on_scores(final_candidates
                                                               )
    return final_candidates


if __name__ == "__main__":
    resume_text_db = load_db(
        "embeddings/resume_classifier/embeddingsBase/chroma/train/Resume")
    sample = rags.get_sample(resume_text_db, 0)
    requirements = sample["sample"]["Category"]
    final_candidates = get_candidates_given_job(requirements, resume_text_db)

    for id_candidate, candidate in final_candidates.items():
        job_role = rags.get_sample(resume_text_db, int(
            id_candidate))["sample"]["Category"]
        print(f"db role label: {job_role}\n{candidate}\n\n")
