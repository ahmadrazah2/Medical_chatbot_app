from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load(self) -> List[Document]:
        """
        Load all PDFs from a directory using DirectoryLoader
        """
        loader = DirectoryLoader(
            path=str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )

        docs = loader.load()

        # Add file source clearly
        for d in docs:
            d.metadata["source"] = Path(d.metadata.get("source", "")).name

        return docs
