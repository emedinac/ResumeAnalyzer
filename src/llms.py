from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# https://python.langchain.com/api_reference/_modules/langchain_huggingface/llms/huggingface_pipeline.html#HuggingFacePipeline


def build_llm(model="tiiuae/falcon-7b", max_new_tokens=512):
    pipe = pipeline("text-generation",
                    model=model,
                    max_new_tokens=max_new_tokens)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
