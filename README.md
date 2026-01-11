

# 🏥 Medical Chatbot — Retrieval-Augmented Generation (RAG)

A **production-ready medical question-answering assistant** built with **Streamlit**, **LangChain**, **ChromaDB**, and **HuggingFace**.
This system uses **Retrieval-Augmented Generation (RAG)** to provide **fact-grounded, reliable answers** from medical literature (`Medical_book.pdf`) instead of hallucinated responses.

This project demonstrates how **LLMs + Vector Databases + Document Retrieval** can be combined into a real-world **AI assistant for healthcare knowledge**.

🌐 Live Demo
👉 [https://medicalchatbotapp.streamlit.app/]
---

## 🚀 Key Features

✅ **RAG-Powered Medical QA**
Retrieves the most relevant medical text before generating an answer, ensuring factual accuracy.

✅ **Local Vector Database (ChromaDB)**
All embeddings are stored locally for **fast, private, and cost-free** retrieval.

✅ **Modern LLM (Mistral-7B-Instruct)**
Uses **Mistral-7B-Instruct-v0.2** via HuggingFace for high-quality responses.

✅ **Efficient English Embeddings**
Uses **`BAAI/bge-small-en-v1.5`** — small, fast, and highly accurate for semantic search.

✅ **ChatGPT-Style UI**
Built with Streamlit, featuring chat history, clean layout, and easy interaction.

✅ **Source Transparency**
Every answer includes the **exact document chunks** used by the model.

---

## 🧠 System Architecture

```
User Query
    ↓
Embedding (bge-small-en)
    ↓
ChromaDB Vector Search
    ↓
Relevant Medical Chunks
    ↓
Mistral-7B Instruct
    ↓
Grounded Medical Answer
```

This prevents hallucinations and ensures all answers come **directly from medical sources**.

---

## 🛠️ Tech Stack

| Component  | Technology                             |
| ---------- | -------------------------------------- |
| Frontend   | Streamlit                              |
| LLM        | Mistral-7B-Instruct-v0.2 (HuggingFace) |
| Embeddings | BAAI/bge-small-en-v1.5                 |
| Vector DB  | ChromaDB                               |
| Framework  | LangChain                              |
| Language   | Python 3.10+                           |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ahmadrazah2/Medical_chatbot_app.git
cd Medical_chatbot_app
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Medical Data

Place your PDF here:

```
data/Medical_book.pdf
```

The vector database will be automatically created on first run.

---

## 🔑 HuggingFace API Setup

This project uses HuggingFace Inference API for the LLM.

### 1️⃣ Get Token

Generate an API key from:
👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 2️⃣ Add to Streamlit Secrets

Create:

```
.streamlit/secrets.toml
```

```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
```

For Streamlit Cloud:
**App Settings → Secrets → Add same key**

---

## 🏃 Run the App

```bash
streamlit run streamlit_app.py
```

Then open your browser and start asking:

* *“What causes diabetes?”*
* *“How is hypertension treated?”*
* *“Explain insulin resistance”*

You’ll see:

* AI response
* Source text used
* Full chat history

---

## 📂 Project Structure

```
Medical_chatbot_app/
│
├── .streamlit/
│   └── secrets.toml
│
├── data/
│   └── Medical_book.pdf
│
├── rag/
│   ├── config.py
│   ├── loader.py
│   ├── cleaner.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── llm.py
│   └── rag_chain.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

* 🏥 Medical knowledge assistant
* 📚 Medical book Q&A
* 🧠 Clinical reference tool
* 🏫 Student study assistant
* 🤖 RAG system demo for AI engineers

---

## 🤝 Contributing

Pull requests are welcome!
You can help by:

* Adding more documents
* Improving UI
* Optimizing retrieval

---


