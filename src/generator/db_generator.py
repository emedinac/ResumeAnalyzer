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

llm_model_names = ["meta-llama/Meta-Llama-3-8B-Instruct",
                   "meta-llama/Llama-3.2-1B-Instruct",
                   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                   #    "mradermacher/Grok_HumanLike_Llama-3.2-3B-Instruct-GGUF"
                   "Qwen/Qwen2-7B-Instruct",
                   "google/gemma-7b",
                   ]

embedding_model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
embedding_models = {name: SentenceTransformer(
    name) for name in embedding_model_names}


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
        self.generators = []
        for model_name, gpu in zip(llm_model_names, [0, 0, 1, 1, 1]):
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
                return_full_text=False,
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
        # Generate dataset using the loaded model
        combined_datasets = {}
        gen_templates = [p for p in prompts.__dict__ if "gen_" in p]
        noise_templates = [p for p in prompts.__dict__ if "noise_" in p]
        format_templates = [p for p in prompts.__dict__ if "format_" in p]
        for idx, db in enumerate(self.dbs):
            for split, data in db.items():
                print(f"{idx}/{len(self.db_names)} - {split} - {db}")
                collected = []
                jds_prompts = []
                gen_prompts = []
                noise_prompts = []
                format_prompts = []
                for resume in tqdm(data["resume"], desc="- structuring"):
                    gen_prompt = np.random.choice(gen_templates)
                    gen_prompts.append(getattr(prompts, gen_prompt))
                    noise_prompt = np.random.choice(noise_templates)
                    noise_prompts.append(getattr(prompts, noise_prompt))
                    format_prompt = np.random.choice(format_templates)
                    format_prompts.append(getattr(prompts, format_prompt))
                    # pipeline
                    jds_prompts.append(
                        f"{gen_prompts[-1]}\n{noise_prompts[-1]}\n{format_prompts[-1]}\nRESUME:\n{resume}ANSWER:\n")

                # generate job description
                job_descriptions = []
                for model in tqdm(self.generators, desc="- model:"):
                    for jd in tqdm(jds_prompts, desc="- m(jobs)"):
                        job_descriptions.append(model.invoke(jd))

                categories = []
                sims_group = []
                summarizations = []
                biases = []
                for jds in tqdm(job_descriptions, desc="- metrics"):
                    # classification 3-class
                    cls = []
                    summ = []
                    metrics_prompts1 = []
                    metrics_prompts2 = []
                    for jd in jds:
                        # reasoning explainations about resume-job match
                        prompt_cls = prompts.evaluation_template.format_map({"resume": resume,
                                                                            "job_description": jd})
                        prompt_cls = f"\n{prompt_cls}\nANSWER:\n"
                        metrics_prompts1.append(prompt_cls)
                        prompt_summ = prompts.system_evaluator_template.format_map({"resume": resume,
                                                                                    "job_description": jd})
                        prompt_summ = f"\n{prompt_summ}\nANSWER:\n"
                        metrics_prompts2.append(prompt_summ)
                    for m in self.generators:
                        cls.append(m.invoke(metrics_prompts1))
                        summ.append(m.invoke(metrics_prompts2))

                    categories.append(cls)
                    summarizations.append(summ)

                    # Compute biases
                    # metrics (compute_similarity) about the resume and job description
                    biases.append([m(jd) for m in self.bias_models])
                    sims = {name: compute_similarity(
                        resume, jd, m) for name, m in embedding_models.items()}
                    sims_group.append(sims)

                    dcscore = compute_dcscore_softmax(jds,
                                                      embedding_models["intfloat/e5-large-v2"]
                                                      )
                    ngram_score = ngram_diversity_score(jds)
                    compress_ratio = compression_ratio(jds)
                    homogenization_rouge = homogenization_score(jds,
                                                                'rougel')

                    homogenization_bertscore = homogenization_score(jds,
                                                                    'bertscore')

                    collected.append({
                        "resume": resume,
                        "job_description": jds,
                        "relevance_label": categories,
                        "explanation": summarizations,
                        "semantic_similarity": sims_group,
                        "biases": biases,
                        "dcscore": dcscore,
                        "ngram_score": ngram_score,
                        "homogenization_rouge": homogenization_rouge,
                        "homogenization_bertscore": homogenization_bertscore,
                        "compress_ratio": compress_ratio,
                        "gen_prompts": gen_prompts,
                        "noise_prompts": noise_prompts,
                        "format_prompts": format_prompts,
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
