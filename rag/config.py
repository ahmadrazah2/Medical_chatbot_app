from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    pdf_path: Path = Path("data") / "Medical_book.pdf"
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "medical_book"

    # embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # llm_path is no longer needed for HuggingFace API

    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 120

    # Retrieval
    k: int = 2

    # HuggingFace
    repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # hf_token can be loaded from environment variable HUGGINGFACEHUB_API_TOKEN

