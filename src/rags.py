
from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

import prompts
import numpy as np
import re


def reorder_candidates_based_on_scores(candidates: dict) -> dict:
    def get_score(metric: dict) -> int:
        return int(metric.get("RECOMMENDED_SCORE", 0))
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda item: get_score(item[1]),
        reverse=True
    )
    return dict(sorted_candidates)


def get_sample(db, idx=None, split="train"):
    if split in db.dataset:
        data = db.dataset[split]
    else:
        data = db.dataset
    if idx is None:
        sample = data[np.random.randint(len(data))]
    else:
        sample = data[idx]
    # job_chunk = db.vectors.similarity_search(sample["query_text"], k=5)
    return {"sample": sample,
            "split": split,
            "idx": idx
            }


class RAGSystem:
    # format_instructions could be the specific outputs from the candidate evaluation
    def __init__(self, db, vectorstore, search_type="similarity", search_kwargs=None,
                 system_template=None, query_template=None, evaluation_template=None,
                 extraction_template=None):
        self.setup(db, vectorstore, search_type, search_kwargs)
        if system_template is None:
            self.system_template = prompts.system_template
        if query_template is None:
            self.query_template = prompts.cv_question1
        if evaluation_template is None:
            self.evaluation_template = prompts.evaluation_template
        if extraction_template is None:
            self.extraction_template = prompts.extraction_template

    def setup(self, db, vectorstore, search_type, search_kwargs):
        self.set_vectorstore(vectorstore)
        self.set_retriever(search_type, search_kwargs)
        self.set_db(db)

    def set_retriever(self, search_type, search_kwargs):
        self.retriever = self.vectorstore.as_retriever(search_type=search_type,
                                                       search_kwargs=search_kwargs,
                                                       )

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    def set_db(self, db):
        self.db = db

    def get_relevant_requeriments(self, query, llm):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.system_template),
            HumanMessagePromptTemplate.from_template(
                self.extraction_template
            ),
        ])
        str_prompt = prompt.format_prompt(
            context=query,
        ).to_string()
        job_requirements = llm.invoke(str_prompt) + ", " + query
        return job_requirements

    def evaluate_cv(self, job_description, resume, llm):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.system_template),
            SystemMessagePromptTemplate.from_template(
                self.query_template
            ),
        ])
        str_prompt = prompt.format_prompt(
            role=job_description,
            context=resume
        ).to_string()
        evaluation = llm.invoke(str_prompt)
        candidate = {"resume": resume, "evaluation": evaluation}
        return candidate

    def get_relevant_cvs(self, query, llm, split="train",):

        docs_chunks = []
        vals, cnts = np.unique(query.split(","), return_counts=True)
        for q in vals:
            docs_chunks.extend(self.retriever.invoke(q))
        docs_chunks.extend(self.retriever.invoke(query))
        docs = {d.metadata["doc_id"] for d in docs_chunks}

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.system_template),
            SystemMessagePromptTemplate.from_template(
                self.query_template
            ),
        ])
        candidates = {}
        for doc_id in docs:
            resume = get_sample(self.db,
                                idx=int(doc_id),
                                split=split)["sample"]['Resume']

            str_prompt = prompt.format_prompt(
                role=str([", ".join(vals)]),
                context=resume
            ).to_string()
            evaluation = llm.invoke(str_prompt)
            candidates[doc_id] = {"resume": resume, "evaluation": evaluation}
        return candidates

    def evaluate_cv_job_pairwise(self, resume, query, metrics, llm):
        # Basically a regex
        str1 = "SCORE:"
        str2 = "POTENTIAL_FIT_AREAS:"
        str3 = "SUMMARY:"
        idx1 = metrics.find(str1)
        idx2 = metrics.find(str2)
        idx3 = metrics.find(str3)
        if idx3 == -1:
            return "INVALID CV ANALYSIS"
        score = metrics[idx1+len(str1):idx2].strip()
        if "." in score or "," in score:
            value = score.strip().split(':')[1] * 100
            score = f"{round(value)}\n"
        fit_areas = metrics[idx2+len(str2):idx3]
        summary = metrics[idx3+len(str3):]
        prompt = self.evaluation_template.format_map({"context": resume,
                                                      "role": query,
                                                      "score": score,
                                                      "summary": summary,
                                                      "fit_areas": fit_areas,
                                                      })
        response = llm.invoke(prompt)
        out = self.filter_answer(response)
        if "no" in out["VALID_SCORE"].lower() or \
            "no" in out["VALID_SUMMARY"].lower() or \
                "no" in out["VALID_FIT_AREAS"].lower():
            return "INCOMPLETE CV ANALYSIS\n"+response
        out["SUMMARY"] = summary
        return out

    @staticmethod
    def filter_answer(response):
        str1 = "- VALID_SCORE:"
        str2 = "- VALID_SUMMARY:"
        str3 = "- VALID_POTENTIAL_FIT_AREAS:"
        str4 = "- RECOMMENDED_CLASS:"
        str5 = "- ERRORS_OR_INCONSISTENCIES:"
        str6 = "- RECOMMENDED_SCORE:"

        idx1 = response.find(str1)
        idx2 = response.find(str2)
        idx3 = response.find(str3)
        idx4 = response.find(str4)
        idx5 = response.find(str5)
        idx6 = response.find(str6)
        VALID_SCORE = response[idx1+len(str1):idx2].strip()
        VALID_SUMMARY = response[idx2+len(str2):idx3].strip()
        VALID_FIT_AREAS = response[idx3+len(str3):idx4].strip()
        RECOMMENDED_CLASS = response[idx4+len(str4):idx5].strip()
        ERRORS_OR_INCONSISTENCIES = response[idx5+len(str5):idx6].strip()
        RECOMMENDED_SCORE = response[idx6+len(str6):].strip()
        return {"VALID_SCORE": VALID_SCORE,
                "VALID_SUMMARY": VALID_SUMMARY,
                "VALID_FIT_AREAS": VALID_FIT_AREAS,
                "RECOMMENDED_CLASS": RECOMMENDED_CLASS,
                "ERRORS_OR_INCONSISTENCIES": ERRORS_OR_INCONSISTENCIES,
                "RECOMMENDED_SCORE": RECOMMENDED_SCORE, }
