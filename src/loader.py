from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
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

    def compute_embeddings(self, split="train", batch_size=1024, num_workers=16):
        # Compute embeddings without chunks
        self.chunks = {}
        fields = list(self.dataset[split].features.keys())
        for field in fields:
            if "label" == field:
                continue
            samples = self.dataset[split][field]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self.chunker.create_documents,
                                    samples[i:i + batch_size])
                    for i in range(0, len(samples), batch_size)
                ]
                chunks = [future.result() for future in futures]
            # chunks = self.chunker.create_documents(samples)
            print(f"Total chunks: {len(chunks)}")
            self.chunks[field] = chunks

    def build_vectorstore(self, vectorstore_type="faiss"):
        self.vectorstore_type = vectorstore_type
        fields = list(self.chunks.keys())
        for field in fields:
            if vectorstore_type == "faiss":
                self.vector_db = FAISS.from_documents(self.chunks[field],
                                                      self.embedding_model,
                                                      )
            elif vectorstore_type == "chroma":
                self.vector_db = Chroma.from_documents(self.chunks[field],
                                                       self.embedding_model,
                                                       )
            else:
                raise ValueError(
                    f"Unknown vectorstore: {vectorstore_type}")

    def load_indexes(self, vectorstore_type="faiss", loade_path="embeddings"):
        load_path = Path(loade_path)
        if vectorstore_type == "faiss":
            self.vector_db = FAISS.load_local(str(load_path),
                                              self.embedding_model,
                                              allow_dangerous_deserialization=True
                                              )
        elif vectorstore_type == "chroma":
            self.vector_db = Chroma(persist_directory=loade_path,
                                    embedding_function=self.embedding_model,
                                    )
        return self.vector_db

    def save_indexes(self, split="train", save_path="embeddings"):
        if self.vectorstore_type == "faiss":
            save_path = Path(save_path).joinpath(self.vectorstore_type, split)
            save_path.mkdir(parents=True, exist_ok=True)
            self.vector_db.save_local(f"{str(save_path)}")
        elif self.vectorstore_type == "chroma":
            save_path = Path(save_path).joinpath(self.vectorstore_type, split)
            save_path.mkdir(parents=True, exist_ok=True)
            self.vector_db.persist(persist_directory=save_path)


if __name__ == "__main__":
    # Example usage
    dataset = ResumeLoader()
    for split in ["train", "test"]:
        print(f"Processing split: {split}")
        if Path("embeddings/{split}").exists():
            dataset.load_indexes(split)
            print("Embeddings loaded successfully.")
        else:
            dataset.compute_embeddings(split)
            dataset.build_vectorstore(vectorstore_type="faiss")
            dataset.save_indexes(split)
            dataset.compute_embeddings(split)
            dataset.build_vectorstore(vectorstore_type="chroma")
            dataset.save_indexes(split)
            print("Embeddings computed and saved successfully.")

    print("Embeddings computed and saved successfully.")
