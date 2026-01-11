import re
from typing import List
from langchain_core.documents import Document

class TextCleaner:
    def __init__(self):
        self.hyphen_fix = re.compile(r"(\w)-\n(\w)")
        self.multi_space = re.compile(r"[ \t]+")
        self.many_newlines = re.compile(r"\n{3,}")

    def clean_docs(self, docs: List[Document]) -> List[Document]:
        cleaned_docs = []

        for d in docs:
            text = d.page_content or ""

            text = self.hyphen_fix.sub(r"\1\2", text)
            text = self.multi_space.sub(" ", text)
            text = self.many_newlines.sub("\n\n", text)

            cleaned_docs.append(
                Document(
                    page_content=text.strip(),
                    metadata=d.metadata
                )
            )

        return cleaned_docs
