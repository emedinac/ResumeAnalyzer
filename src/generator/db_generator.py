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

torch.set_float32_matmul_precision('high')

llm_model_names = ["tiiuae/Falcon3-1B-Instruct",
                   "meta-llama/Llama-3.2-1B-Instruct",
                   #    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                   #    "Qwen/Qwen2.5-1.5B-Instruct",
                   "google/gemma-3-1b-it",
                   ]

embedding_model_names = [
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
embedding_models = {name: SentenceTransformer(
    name) for name in embedding_model_names}


def compute_similarity(resume, jd, model):
    with torch.no_grad():
        emb1 = model.encode(resume, convert_to_tensor=True)
        emb2 = model.encode(jd, convert_to_tensor=True)
        emb1 = emb1 / emb1.norm(p=2)
        emb2 = emb2 / emb2.norm(p=2)
        return util.cos_sim(emb1, emb2).item()  # cosine similarity/distance :)


def compute_dcscore_softmax(texts, model, temp=0.1):
    with torch.no_grad():
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
        # Prepare prompt templates once
        self.gen_templates = [getattr(prompts, p)
                              for p in prompts.__dict__ if p.startswith("gen_")]
        self.noise_templates = [
            getattr(prompts, p) for p in prompts.__dict__ if p.startswith("noise_")]
        self.fmt_templates = [
            getattr(prompts, p) for p in prompts.__dict__ if p.startswith("format_")]

        # LLM for JOB Generation
        self.generators = []
        for model_name, gpu in zip(llm_model_names, [0, 0, 0, 0, 0]):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            print(gpu, model_name)
            cv_pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                batch_size=8,
                temperature=0.5,
                top_p=0.95,
                max_new_tokens=1024,
                return_full_text=False,
                torch_dtype=torch.float16,
                device=gpu
            )
            self.generators.append(HuggingFacePipeline(pipeline=cv_pipe))

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
        with torch.no_grad():
            # Generate dataset using the loaded model
            combined_datasets = {}
            for idx, db in enumerate(self.dbs):
                for split, data in db.items():
                    print(f"{idx}/{len(self.db_names)} - {split} - {db}")
                    collected = []
                    resumes = data["resume"]
                    for ii, resume in tqdm(enumerate(resumes),
                                           desc="- structuring",
                                           total=len(resumes)
                                           ):
                        gen_prompt = str(
                            np.random.choice(self.gen_templates))
                        noise_prompt = str(
                            np.random.choice(self.noise_templates))
                        format_prompt = str(
                            np.random.choice(self.fmt_templates))

                        # pipeline
                        prompt_jd = f"{gen_prompt}\n{noise_prompt}\n{format_prompt}\nRESUME:\n{resume}ANSWER:\n"
                        job_descriptions = []
                        categories = []
                        sims_group = []
                        summarizations = []
                        biases = []
                        for model in tqdm(self.generators, desc="- model"):
                            # generate job description
                            jd = model.invoke(prompt_jd)
                            job_descriptions.append(jd)

                            # classification 3-class and reasoning explainations about resume-job match
                            cls = []
                            summ = []
                            for _ in range(len(self.generators)):
                                prompt_cls = prompts.evaluation_template.format_map({"resume": resume,
                                                                                    "job_description": jd})
                                cls.append(f"\n{prompt_cls}\nANSWER:\n")
                                prompt_summ = prompts.system_evaluator_template.format_map({"resume": resume,
                                                                                            "job_description": jd})
                                summ.append(f"\n{prompt_summ}\nANSWER:\n")

                            for m in tqdm(self.generators, desc="- metrics"):
                                categories.append(m.invoke(cls))
                                summarizations.append(m.invoke(summ))

                            # Compute biases
                            biases.append([m(jd) for m in self.bias_models])

                            # metrics (compute_similarity) about the resume and job description
                            sims = {name: compute_similarity(
                                resume, jd, m) for name, m in embedding_models.items()}
                            sims_group.append(sims)

                        print()
                        # More Metrics
                        dcscore = compute_dcscore_softmax(job_descriptions,
                                                          embedding_models["sentence-transformers/all-mpnet-base-v2"]
                                                          )
                        ngram_score = ngram_diversity_score(job_descriptions)
                        compress_ratio = compression_ratio(job_descriptions)
                        homogenization_rouge = homogenization_score(job_descriptions,
                                                                    'rougel')

                        homogenization_bertscore = homogenization_score(job_descriptions,
                                                                        'bertscore')

                        obj = {
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
                            "gen_prompt": gen_prompt,
                            "noise_prompt": noise_prompt,
                            "format_prompt": format_prompt,
                            "db_name": self.db_names[idx],
                            "split": split,
                        }
                        np.save(
                            f"db/collected_{idx:05d}_{split}_{ii:05d}.npy", obj)
                        collected.append(obj)
                    ds_split = Dataset.from_list(collected)
                    if split in combined_datasets:
                        combined_datasets[f"{idx}_{split}"] = concatenate_datasets([combined_datasets[f"{idx}_{split}"],
                                                                                    ds_split])
                    else:
                        combined_datasets[f"{idx}_{split}"] = ds_split
                    np.save(f"collected_{idx}_{split}.npy", collected)
            final_dataset = DatasetDict(combined_datasets)
            final_dataset.save_to_disk("resume-job-match")
            print("\n\nDataset Completed!!\n\n")


if __name__ == "__main__":
    generator = ResumeJobMatchGenerator()
    generator.generate()
