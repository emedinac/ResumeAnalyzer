from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# looks like the best in the 0.3v
# https://medium.com/@anixlynch/7-chunking-strategies-for-langchain-b50dac194813
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.schema import Document
from tqdm import tqdm
import re  # used to add spaces for links, dates and numbers


import re


def normalize_resume_text(text):
    return text


class BaseResumeLoader:
    """Base Loader class to load the resume-job-description-fit dataset"""

    def __init__(self, split="train",
                 batch_size=256,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 ):
        # Another one interesting: d4rk3r/resumes-raw-pdf
        self.split = split
        self.model_name = model_name
        self.batch_size = batch_size

    def setup(self, keep_in_memory=True):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={"normalize_embeddings": True,
                           "convert_to_tensor": True,
                           "batch_size": self.batch_size, },
            # multi_process=True,
        )
        db1 = load_dataset("ahmedheakl/resume-atlas",
                           keep_in_memory=keep_in_memory,
                           split=self.split)
        db2 = load_dataset("Lakshmi12/Resume_Dataset",
                           keep_in_memory=keep_in_memory,
                           split=self.split)
        db1 = db1.rename_columns({
            "Text": "Resume",
        })
        db2 = db2.rename_columns({
            "Resume_str": "Resume",
        })
        db1 = db1.select_columns(["Resume", "Category"])
        db2 = db2.select_columns(["Resume", "Category"])
        self.dataset = concatenate_datasets([db1, db2])
        self.split = None
        self.fields = None
        # https://www.rohan-paul.com/p/document-digitization-and-chunking
        # They found ~550 tokens per chunk yielded the best retrieval performance in their setting,
        # balancing context and specificity.
        # Smaller chunks (e.g. 450 tokens) or larger (650+) underperformed slightly.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "•", ":", "\t", ".", "- ", " ", ""]
        )
        self.chunker = SemanticChunker(
            embeddings=self.embedding_model,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75,  # or 90 with chunk_size=1000 and chunk_overlap=100
            sentence_split_regex=r"(?<=[.?!•])\s*",  # .?!
        )

    def compute_embeddings(self, split="train"):
        # Compute embeddings without chunks
        self.split = split
        self.db_chunks = {}
        self.fields = list(self.dataset.features.keys())
        for field in self.fields:
            if "Category" == field:
                continue
            samples = self.dataset[field]  # [:2]  # for tests
            # chunks = self.chunker.create_documents(samples,)
            # This implementation is more accurate and got better performance:
            docs: list[Document] = []
            for doc_idx, sample in enumerate(tqdm(samples, desc="Chunking resumes")):
                # Ensures chunks stay under size limits and preserve natural language flow
                sample = normalize_resume_text(sample)
                prelim_docs = self.text_splitter.split_text(sample)
                chunks = self.chunker.create_documents(prelim_docs)
                # clean chunks
                chunks = [chunk for chunk in chunks if chunk.page_content.strip(
                ) and chunk.page_content.strip() != "."]
                # adding metadata to every chunk manually
                for idx, chunk in enumerate(chunks):
                    chunk.metadata = {"doc_id": str(doc_idx),
                                      "field": str(field),
                                      "split": str(split),
                                      "chunk_index": str(idx),
                                      }
                docs.extend(chunks)
                break
            print(f"Total chunks: {len(docs)}")
            self.db_chunks[field] = docs


class ResumeLoaderFAISS(BaseResumeLoader):
    def __init__(self,
                 split="train",
                 batch_size=256,
                 model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__(split, batch_size, model_name)
        self.vectors = {}

    def build_vectorstore(self):
        for field in self.db_chunks.keys():
            self.vectors[field] = FAISS.from_documents(self.db_chunks[field],
                                                       self.embedding_model,
                                                       )

    def load_indexes(self, loade_path="embeddings"):
        self.vectors = FAISS.load_local(loade_path,
                                        self.embedding_model,
                                        allow_dangerous_deserialization=True
                                        )
        return self.vectors

    def save_indexes(self, save_path="embeddings"):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for field in self.db_chunks.keys():
            db_path = str(save_path.joinpath(field))
            self.vectors[field].save_local(f"{db_path}")


class ResumeLoaderChroma(BaseResumeLoader):
    def __init__(self,
                 split="train",
                 batch_size=256,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 keep_in_memory=True):
        super().__init__(split, batch_size, model_name)
        super().setup(keep_in_memory)
        self.vectors = None

    def build_vectorstore(self, save_path):
        for field in self.db_chunks.keys():
            db_path = Path(save_path).joinpath(field)
            db_path.mkdir(parents=True, exist_ok=True)
            self.vectors = Chroma.from_documents(self.db_chunks[field],
                                                 self.embedding_model,
                                                 # horrible from Chroma
                                                 persist_directory=str(
                db_path),
            )

    def load_indexes(self, loade_path="embeddings"):
        self.vectors = Chroma(persist_directory=loade_path,
                              embedding_function=self.embedding_model,
                              )
        return self.vectors

    def save_indexes(self):
        NotImplemented  # Chroma saves automatically to the persist_directory
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='loader', description='Get and process the dataset')
    parser.add_argument(
        '--model', default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        '--input_path', type=str, default="embeddings")
    parser.add_argument(
        '--output_path', type=str, default="embeddings")
    parser.add_argument(
        '--batch_size', type=int, default=256)
    parser.add_argument(
        '--split', type=str, default="train")
    args = parser.parse_args()
    # "sentence-transformers/all-MiniLM-L6-v2"
    # "sentence-transformers/all-mpnet-base-v2"

    # TEST: Example usage, in Practice, it should be in a separate script
    field = "resume_text"
    dataset_faiss = ResumeLoaderFAISS(batch_size=args.batch_size,
                                      model_name=args.model)
    dataset_chroma = ResumeLoaderChroma(batch_size=args.batch_size,
                                        model_name=args.model)
    split = args.split
    print(f"Processing split: {split}")
    if Path(f"embeddings/chroma/{split}/{field}").exists():
        dataset_chroma.load_indexes(
            f"{args.input_path}/chroma/{split}/{field}")
        dataset_faiss.load_indexes(f"{args.input_path}/faiss/{split}/{field}")
        print("Embeddings loaded successfully.")
    else:
        dataset_faiss.setup()
        dataset_faiss.compute_embeddings(split)
        dataset_faiss.build_vectorstore()
        dataset_faiss.save_indexes(f"{args.output_path}/faiss/{split}")
        dataset_chroma.db_chunks = dataset_faiss.db_chunks
        dataset_chroma.build_vectorstore(f"{args.output_path}/chroma/{split}")
        print("Embeddings computed and saved successfully.")

    print("Computation finished successfully.")
