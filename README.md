# LangChain RAG Toolkit

> **A modular LangChain project showcasing Retrieval-Augmented Generation (RAG), translation chains, and conversational memory with visual assets.**

---

## 📌 Overview

This repository demonstrates **practical LangChain implementations** using OpenAI models. It contains three core Python scripts that highlight different functionalities of the LangChain framework:

1. **`RAG.py`** – Retrieval-Augmented Generation with structured Markdown output.
2. **`simplemessage.py`** – Language translation chain with FastAPI serving.
3. **`SimpleChatwithMemory.py`** – Stateful conversational agent with in-memory chat history.

Additionally, the repo includes an `Assets/` folder containing explanatory figures (a1–a5) that illustrate how the system works.

---

## 📂 Project Structure

```bash
├── Assets/
│   ├── a1.png   # RAG pipeline illustration
│   ├── a2.png   # Chunking & embeddings visualization
│   ├── a3.png   # Markdown rendering example
│   ├── a4.png   # Translation API workflow
│   └── a5.png   # Chat with memory workflow
│
├── RAG.py
├── simplemessage.py
├── SimpleChatwithMemory.py
└── requirements.txt
```

---

## 🚀 Implementations

### 1. Retrieval-Augmented Generation – `RAG.py`

**Key Features:**

* Web scraping with **BeautifulSoup (bs4)**.
* Document chunking using **RecursiveCharacterTextSplitter**.
* Vector storage and retrieval via **Chroma**.
* Markdown-formatted responses powered by **ChatPromptTemplate**.
* Beautiful terminal rendering with **Rich**.

📷 *Illustrations:* See `Assets/a1.png`, `Assets/a2.png`, `Assets/a3.png`.

---

### 2. Translation Chain API – `simplemessage.py`

**Key Features:**

* Dynamic translation into any specified language.
* Prompting with **ChatPromptTemplate**.
* Serving via **FastAPI** + **LangServe**.
* Endpoint available at: `http://localhost:8000/chain`

📷 *Illustration:* See `Assets/a4.png`.

---

### 3. Conversational Memory – `SimpleChatwithMemory.py`

**Key Features:**

* Maintains dialogue history across multiple turns.
* Uses **InMemoryChatMessageHistory** for session persistence.
* Configurable session IDs for multiple user contexts.
* Streamed responses for real-time conversation.

📷 *Illustration:* See `Assets/a5.png`.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/LangChain-RAG-Toolkit.git
cd LangChain-RAG-Toolkit
pip install -r requirements.txt
```

Make sure you have a valid **OpenAI API key** stored in a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Usage

### Run RAG example

```bash
python RAG.py
```

### Run Translation API

```bash
uvicorn simplemessage:app --reload
```

Visit [http://localhost:8000/chain](http://localhost:8000/chain)

### Run Chat with Memory

```bash
python SimpleChatwithMemory.py
```

---

## 📊 Requirements

The project relies on the following dependencies (see `requirements.txt` for exact versions):

* **LangChain ecosystem:** `langchain`, `langchain-core`, `langchain-openai`, `langchain-community`, `langchain-chroma`, `langchainhub`
* **LLM API:** `openai`
* **Web scraping & parsing:** `bs4`
* **Vector store & utilities:** `faiss-cpu`, `tqdm`, `tiktoken`
* **Server:** `fastapi`, `langserve`, `sse_starlette`
* **Environment & styling:** `python-dotenv`, `rich`

---

## 🎯 Goals

This project is designed for:

* Learning **retrieval-augmented generation (RAG)**.
* Building and deploying **LangChain pipelines**.
* Demonstrating **translation services**.
* Exploring **chatbots with memory**.

---

## 📖 References

* [LangChain Documentation](https://python.langchain.com/)
* [OpenAI API Reference](https://platform.openai.com/docs/)
* [Chroma Documentation](https://docs.trychroma.com/)

---

## 📝 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.

---

## 🙌 Acknowledgments

Special thanks to the open-source community and the LangChain contributors for making these tools available.
