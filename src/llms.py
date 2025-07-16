from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, GenerationConfig

# https://python.langchain.com/api_reference/_modules/langchain_huggingface/llms/huggingface_pipeline.html#HuggingFacePipeline


class LLMScorer:
    def __init__(self, model, temperature=0.1, max_new_tokens=512):
        tokenizer = AutoTokenizer.from_pretrained(model)

        cv_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        self.analyzer = HuggingFacePipeline(pipeline=cv_pipe)

        judge_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        self.judge = HuggingFacePipeline(pipeline=judge_pipe)
