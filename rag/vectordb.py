from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def split(self, docs: List[Document], chunk_size: int, overlap: int):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        return splitter.split_documents(docs)

    def build_or_load(self, docs, embeddings):
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            db = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=embeddings,
            )
            if db._collection.count() > 0:
                return db
        except Exception:
            pass

        db = Chroma.from_documents(
            docs,
            embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
        )
        db.persist()
        return db
