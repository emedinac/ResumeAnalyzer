from loader import ResumeLoader
from pathlib import Path


dataset = ResumeLoader()
for split in ["train", "test"]:
    if Path("embeddings/{split}").exists():
        dataset.load_embeddings(split)
        print("Embeddings loaded successfully.")
    else:
        dataset.compute_embeddings(split)
        dataset.save_embeddings(split)
        print("Embeddings computed and saved successfully.")
