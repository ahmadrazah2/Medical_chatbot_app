from typing import List, Tuple
from langchain_core.documents import Document

PROMPT = """You are a medical assistant for question-answering tasks. "
    "Use ONLY the retrieved context to answer. "
    "If the retrieved context is empty or does not contain the answer, say you don't know. "
    "Answer in English. "
    "Do not invent information outside the provided context. "
    "Keep answers concise (max 3 sentences).\n\n"
Context:
{context}

Question:
{question}

Answer:
"""

class RAGPipeline:
    def __init__(self, vectordb, llm, k: int):
        self.vectordb = vectordb
        self.llm = llm
        self.k = k
        self.max_context_chars = 3000

    def retrieve(self, query: str) -> List[Document]:
        retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})
        return retriever.invoke(query)

    def _format_context(self, docs: List[Document]) -> str:
        # keep context short and clean
        blocks: List[str] = []
        total = 0
        for i, d in enumerate(docs, start=1):
            text = (d.page_content or "").strip()
            if text:
                block = f"[{i}] {text}"
                total += len(block)
                if total > self.max_context_chars:
                    break
                blocks.append(block)
        return "\n\n".join(blocks)

    def answer(self, question: str) -> Tuple[str, List[Document]]:
        docs = self.retrieve(question)
        context = self._format_context(docs)
        prompt = PROMPT.format(context=context, question=question)
        out = self.llm.invoke(prompt)
        # ChatHuggingFace returns an AIMessage, extract the content
        answer_text = out.content if hasattr(out, "content") else str(out)
        return answer_text.strip(), docs
