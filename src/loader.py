from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# looks like the best in the 0.3v
# https://medium.com/@anixlynch/7-chunking-strategies-for-langchain-b50dac194813
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

from pathlib import Path


class ResumeLoader:
    """Loader class to load the resume-job-description-fit dataset."""

    def __init__(self, path_to_dataset="cnamuangtoun/resume-job-description-fit"):
        # Another one interesting: d4rk3r/resumes-raw-pdf
        self.dataset = load_dataset(path_to_dataset)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_db = None
        self.embeddings = {}

        self.chunker = SemanticChunker(
            embeddings=self.embedding_model,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
            sentence_split_regex=r"(?<=[.?!])\s+",  # .?!
        )

    def compute_embeddings_faiss(self, split="train", batch_size=1024):
        # Compute embeddings without chunks
        fields = list(self.dataset[split].features.keys())
        for field in fields:
            if "label" == field:
                continue
            all_chunks = []
            batches = (len(self.dataset[split]) // batch_size)+1
            for i in tqdm(range(batches)):
                samples = self.dataset[split][field][i *
                                                     batch_size:
                                                     (i+1) *
                                                     batch_size]
                chunks = self.chunker.create_documents(samples)
                all_chunks.extend(chunks)
            print(f"Total chunks: {len(all_chunks)}")
            self.embeddings[field] = FAISS.from_documents(all_chunks,
                                                          self.embedding_model)

    def load_embeddings_faiss(self, split="train", loade_path="embeddings"):
        load_path = Path(loade_path).joinpath(split)
        if not load_path.exists():
            raise FileNotFoundError(f"Path {str(load_path)} does not exist.")
        self.vector_db = FAISS.load_local(str(load_path),
                                          self.embedding_model,
                                          allow_dangerous_deserialization=True
                                          )
        return self.vector_db

    def save_embeddings_faiss(self, split="train", save_path="embeddings"):
        save_path = Path(save_path).joinpath(split)
        save_path.mkdir(parents=True, exist_ok=True)
        self.embeddings.save_local(f"{str(save_path)}")


if __name__ == "__main__":
    # Example usage
    dataset = ResumeLoader()
    for split in ["train", "test"]:
        print(f"Processing split: {split}")
        if Path("embeddings/{split}").exists():
            dataset.load_embeddings_faiss(split)
            print("Embeddings loaded successfully.")
        else:
            dataset.compute_embeddings_faiss(split)
            dataset.save_embeddings_faiss(split)
            print("Embeddings computed and saved successfully.")

    print("Embeddings computed and saved successfully.")
