
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


def get_sample(db, idx=None, split="train"):
    data = db.dataset[split]
    if idx is None:
        sample = data[np.random.randint(len(data))]
    else:
        sample = data[idx]
    # job_chunk = db.vectors.similarity_search(sample["job_description_text"], k=5)
    return {"sample": sample,
            "split": split,
            "idx": idx
            }


class RAGSystem:
    # format_instructions could be the specific outputs from the candidate evaluation
    def __init__(self, db, vectorstore, search_type="similarity", k=5,
                 system_template=None, query_template=None, evaluation_template=None):
        self.setup(db, vectorstore, search_type, k)
        self.qa_chain = None
        if system_template is None:
            self.system_template = prompts.system_template
        if query_template is None:
            self.query_template = prompts.cv_question1
        if evaluation_template is None:
            self.evaluation_template = prompts.evaluation_template

    def setup(self, db, vectorstore, search_type, k):
        self.set_vectorstore(vectorstore)
        self.set_retriever(search_type=search_type, k=k)
        self.set_db(db)

    def set_retriever(self, search_type="similarity", k=5):
        self.retriever = self.vectorstore.as_retriever(k=k,
                                                       search_type=search_type)

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    def set_db(self, db):
        self.db = db

    def get_relevant_cvs(self, job_description, llm, split="train",):
        docs_chunks = self.retriever.invoke(job_description)
        docs = {d.metadata["doc_id"] for d in docs_chunks}

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.system_template),
            HumanMessagePromptTemplate.from_template(
                self.query_template
            ),
        ])
        candidates = {}
        for doc_id in docs:
            resume = get_sample(self.db,
                                idx=int(doc_id),
                                split=split)["sample"]['resume_text']

            str_prompt = prompt.format_prompt(
                context=job_description,
                input=resume
            ).to_string()
            score = llm.invoke(str_prompt)
            candidates[doc_id] = {"resume": resume, "score": score}
        return candidates

    def evaluate_cv_job_pairwise(self, resume, job_description, llm):
        prompt = self.evaluation_template.format_map({"resume": resume,
                                                     "job_description": job_description
                                                      })
        return llm.invoke(prompt)
