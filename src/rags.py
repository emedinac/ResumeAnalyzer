
from langchain.chains import RetrievalQA
import prompts


def evaluate_cv_with_offer(qa_chain, job_description, prompt_template=prompts.template1):
    prompt = prompt_template.format(job_description=job_description)
    result = qa_chain.invoke(prompt)
    return result


def build_rag(vectorstore, llm, search_type="similarity", k=5):
    retriever = vectorstore.as_retriever(search_type=search_type, k=k)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=retriever,
                                           return_source_documents=True
                                           )
    return qa_chain
