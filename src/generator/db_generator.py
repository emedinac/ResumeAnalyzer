# Inspirations
# https://opus4i.com/syntethic-fine-tuning-data-generation
# https://vlad-mihet.medium.com/addressing-bias-in-ai-prompt-engineering-challenges-and-solutions-4ac51d7584ee
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from langchain_huggingface import HuggingFacePipeline
import prompts
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from diversity import compression_ratio, ngram_diversity_score, homogenization_score
import torch
import torch.nn.functional as F

model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
embedding_models = {name: SentenceTransformer(name) for name in model_names}


def compute_similarity(resume, jd, model):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    emb1 = emb1 / emb1.norm(p=2)
    emb2 = emb2 / emb2.norm(p=2)
    return util.cos_sim(emb1, emb2).item()  # cosine similarity/distance :)


def compute_dcscore_softmax(texts, model, temp=0.1):
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.T
    sim = sim / temp
    softmax_mat = F.softmax(sim, dim=1)
    return softmax_mat.diagonal().mean().item()


class ResumeJobMatchGenerator:
    """Base Loader class to load the resume-job-description-fit dataset"""

    def __init__(self):
        # Dataset merge. Please follow individual license
        self.db_names = ["cnamuangtoun/resume-job-description-fit",
                         "AzharAli05/Resume-Screening-Dataset",
                         "ahmedheakl/resume-atlas",
                         "Manirathinam21/Resume_classification",]
        db1 = load_dataset(self.db_names[0])
        db2 = load_dataset(self.db_names[1])
        db3 = load_dataset(self.db_names[2])
        db4 = load_dataset(self.db_names[3])

        db1 = db1.rename_columns({
            "resume_text": "resume",
            "job_description_text": "job_description",
        })
        db2 = db2.rename_columns({
            "Resume": "resume",
            "Job_Description": "job_description",
        })
        db3 = db3.rename_columns({
            "Text": "resume",
        })
        db4 = db4.rename_columns({
            "text": "resume",
        })
        self.dbs = [db1, db2, db3, db4]

        # LLM for JOB Generation
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        # model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        cv_pipe = pipeline(
            "text-generation",
            batch_size=32,
            model=model_name,
            tokenizer=tokenizer,
            temperature=0.5,
            top_p=0.95,
            return_full_text=False
        )
        generator1 = HuggingFacePipeline(pipeline=cv_pipe)

        model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        cv_pipe = pipeline(
            "text-generation",
            model=model_name,
            batch_size=32,
            tokenizer=tokenizer,
            temperature=0.5,
            top_p=0.95,
            return_full_text=False
        )
        generator2 = HuggingFacePipeline(pipeline=cv_pipe)

        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        cv_pipe = pipeline(
            "text-generation",
            model=model_name,
            batch_size=32,
            tokenizer=tokenizer,
            temperature=0.5,
            top_p=0.95,
            return_full_text=False
        )
        generator3 = HuggingFacePipeline(pipeline=cv_pipe)

        model_name = "xai-org/grok-3-8b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        cv_pipe = pipeline(
            "text-generation",
            model=model_name,
            batch_size=32,
            tokenizer=tokenizer,
            temperature=0.5,
            top_p=0.95,
            return_full_text=False
        )
        generator4 = HuggingFacePipeline(pipeline=cv_pipe)

        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        cv_pipe = pipeline(
            "text-generation",
            model=model_name,
            batch_size=32,
            tokenizer=tokenizer,
            temperature=0.5,
            top_p=0.95,
            return_full_text=False
        )
        generator5 = HuggingFacePipeline(pipeline=cv_pipe)

        self.generators = [generator1,
                           generator2,
                           generator3,
                           generator4,
                           generator5
                           ]

        # BIAS MODELS :)
        bias_model = "maximuspowers/bias-detection-ner"
        tokenizer = AutoTokenizer.from_pretrained(bias_model)
        model = AutoModelForTokenClassification.from_pretrained(bias_model)
        bias_model1 = pipeline("ner", model=model, tokenizer=tokenizer)

        bias_model = "cirimus/modernbert-large-bias-type-classifier"
        tokenizer = AutoTokenizer.from_pretrained(bias_model)
        model = AutoModelForTokenClassification.from_pretrained(bias_model)
        bias_model2 = pipeline("ner", model=model, tokenizer=tokenizer)
        self.bias_models = [bias_model1, bias_model2]

    def generate(self):
        # Generate dataset using the loaded model
        combined_datasets = {}
        gen_templates = [p for p in prompts.__dict__ if "gen_" in p]
        noise_templates = [p for p in prompts.__dict__ if "noise_" in p]
        format_templates = [p for p in prompts.__dict__ if "format_" in p]
        for idx, db in enumerate(self.dbs):
            for split, data in db.items():
                print(f"{idx}/{len(self.db_names)} - {split} - {db}")
                collected = []
                for resume in tqdm(data["resume"]):
                    gen_prompt = np.random.choice(gen_templates)
                    gen_prompt = getattr(prompts, gen_prompt)
                    noise_prompt = np.random.choice(noise_templates)
                    noise_prompt = getattr(prompts, noise_prompt)
                    format_prompt = np.random.choice(format_templates)
                    format_prompt = getattr(prompts, format_prompt)

                    # pipeline
                    prompt_jd = f"{gen_prompt}\n{noise_prompt}\n{format_prompt}\nRESUME:\n{resume}ANSWER:\n"
                    job_descriptions = []
                    categories = []
                    sims_group = []
                    summarizations = []
                    biases = []
                    for _, model in enumerate(self.generators):
                        # generate job description
                        jd = model.invoke(prompt_jd)
                        job_descriptions.append(jd)

                        # classification 3-class
                        cls = []
                        summ = []
                        for m in self.generators:
                            prompt_cls = prompts.evaluation_template.format_map({"resume": resume,
                                                                                "job_description": jd})
                            prompt_cls = f"\n{prompt_cls}\nANSWER:\n"
                            cls.append(m.invoke(prompt_cls))

                            # reasoning explainations about resume-job match
                            prompt_summ = prompts.system_evaluator_template.format_map({"resume": resume,
                                                                                        "job_description": jd})
                            prompt_summ = f"\n{prompt_summ}\nANSWER:\n"
                            summ.append(m.invoke(prompt_summ))
                        categories.append(cls)
                        summarizations.append(summ)

                        # Compute biases
                        biases.append([m(jd) for m in self.bias_models])

                        # metrics (compute_similarity) about the resume and job description
                        sims = {name: compute_similarity(
                            resume, jd, m) for name, m in embedding_models.items()}
                        sims_group.append(sims)

                    # More Metrics
                    dcscore = compute_dcscore_softmax(job_descriptions,
                                                      embedding_models["intfloat/e5-large-v2"]
                                                      )
                    ngram_score = ngram_diversity_score(job_descriptions)
                    compress_ratio = compression_ratio(job_descriptions)
                    homogenization_rouge = homogenization_score(job_descriptions,
                                                                'rougel')

                    homogenization_bertscore = homogenization_score(job_descriptions,
                                                                    'bertscore')

                    collected.append({
                        "resume": resume,
                        "job_description": job_descriptions,
                        "relevance_label": categories,
                        "explanation": summarizations,
                        "semantic_similarity": sims_group,
                        "biases": biases,
                        "dcscore": dcscore,
                        "ngram_score": ngram_score,
                        "homogenization_rouge": homogenization_rouge,
                        "homogenization_bertscore": homogenization_bertscore,
                        "compress_ratio": compress_ratio,
                        "db_name": self.db_names[idx],
                        "split": split,
                    })
                ds_split = Dataset.from_list(collected)
                if split in combined_datasets:
                    combined_datasets[f"{idx}_{split}"] = concatenate_datasets([combined_datasets[f"{idx}_{split}"],
                                                                                ds_split])
                else:
                    combined_datasets[f"{idx}_{split}"] = ds_split
        final_dataset = DatasetDict(combined_datasets)
        final_dataset.save_to_disk("resume-job-match")
        print("\n\nDataset Completed!!\n\n")


if __name__ == "__main__":
    generator = ResumeJobMatchGenerator()
    generator.generate()
