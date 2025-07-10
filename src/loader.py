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


class BaseResumeLoader:
    """Base Loader class to load the resume-job-description-fit dataset"""

    def __init__(self, path_to_dataset="cnamuangtoun/resume-job-description-fit"):
        # Another one interesting: d4rk3r/resumes-raw-pdf
        self.dataset = load_dataset(path_to_dataset)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True},
            # multi_process=True,
        )
        self.split = None
        self.fields = None
        self.chunker = SemanticChunker(
            embeddings=self.embedding_model,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
            sentence_split_regex=r"(?<=[.?!])\s+",  # .?!
        )

    def compute_embeddings(self, split="train"):
        # Compute embeddings without chunks
        self.split = split
        self.chunks = {}
        self.fields = list(self.dataset[split].features.keys())
        for field in self.fields:
            if "label" == field:
                continue
            samples = self.dataset[split][field]  # [:10] # for tests
            chunks = self.chunker.create_documents(samples,)
            print(f"Total chunks: {len(chunks)}")
            self.chunks[field] = chunks


class ResumeLoaderFAISS(BaseResumeLoader):
    def __init__(self, path_to_dataset="cnamuangtoun/resume-job-description-fit"):
        super().__init__(path_to_dataset)
        self.vector_db = {}

    def build_vectorstore(self):
        for field in self.chunks.keys():
            self.vector_db[field] = FAISS.from_documents(self.chunks[field],
                                                         self.embedding_model,
                                                         )

    def load_indexes(self, loade_path="embeddings"):
        self.vector_db = FAISS.load_local(loade_path,
                                          self.embedding_model,
                                          allow_dangerous_deserialization=True
                                          )
        return self.vector_db

    def save_indexes(self, save_path="embeddings"):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for field in self.chunks.keys():
            db_path = str(save_path.joinpath(field))
            self.vector_db[field].save_local(f"{db_path}")


class ResumeLoaderChroma(BaseResumeLoader):
    def __init__(self, path_to_dataset="cnamuangtoun/resume-job-description-fit"):
        super().__init__(path_to_dataset)
        self.vector_db = None

    def build_vectorstore(self, save_path):
        for field in self.chunks.keys():
            db_path = Path(save_path).joinpath("chroma", field)
            db_path.mkdir(parents=True, exist_ok=True)
            self.vector_db = Chroma.from_documents(self.chunks[field],
                                                   self.embedding_model,
                                                   # horrible from Chroma
                                                   persist_directory=str(
                                                       db_path),
                                                   )

    def load_indexes(self, loade_path="embeddings"):
        self.vector_db = Chroma(persist_directory=loade_path,
                                embedding_function=self.embedding_model,
                                )
        return self.vector_db

    def save_indexes(self):
        NotImplemented  # Chroma saves automatically to the persist_directory
        pass


if __name__ == "__main__":
    # Example usage
    field = "resume_text"
    dataset_faiss = ResumeLoaderFAISS()
    dataset_chroma = ResumeLoaderChroma()
    for split in ["train", "test"]:
        print(f"Processing split: {split}")
        if Path(f"embeddings/chroma/{split}/{field}").exists():
            dataset_chroma.load_indexes(f"embeddings/chroma/{split}/{field}")
            dataset_faiss.load_indexes(f"embeddings/faiss/{split}/{field}")
            print("Embeddings loaded successfully.")
            break
        else:
            dataset_faiss.compute_embeddings(split)
            dataset_faiss.build_vectorstore()
            dataset_faiss.save_indexes(f"embeddings/faiss/{split}")
            dataset_chroma.chunks = dataset_faiss.chunks
            dataset_chroma.build_vectorstore("embeddings")
            print("Embeddings computed and saved successfully.")

    print("Computation finished successfully.")
