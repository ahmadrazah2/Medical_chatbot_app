from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def build(self) -> HuggingFaceEmbeddings:
        # Use local model execution with pre-downloaded weights
        cache_path = Path("models") / "hf"
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=str(cache_path),
        )
