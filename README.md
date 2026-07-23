# Local RAG Chat

A 100% local, privacy-focused Retrieval-Augmented Generation (RAG) application built with **Gradio**, **LangChain**, **Qdrant**, and **Ollama**.

Extract content from PDFs (with optional multi-modal vision analysis for charts, formulas, schematics, and portraits) and TXT files, store vectors in memory using hybrid search, and chat with your documents completely offline.

---

## ✨ Key Features

* **100% Local & Private**: No API keys or cloud services required. Everything runs locally via Ollama and an in-memory Qdrant vector store.
* **Multi-Modal Vision Analysis**: Toggle Vision OCR to analyze images, patent drawings, mathematical formulas, schematics, and graphs embedded within PDFs.
* **Markdown Page Parsing**: Uses `pymupdf4llm` to convert PDF layouts into structured, LLM-friendly markdown documents.
* **Hybrid Search Retrieval**: Combines dense vector embeddings (`embeddinggemma`) with sparse BM25 text search (`FastEmbedSparse`) for high-accuracy retrieval.
* **Interactive Gradio UI**:
  * Customizable system roles and prompts.
  * Adjust retrieval parameters on the fly (`k`, `fetch_k`, `score_threshold`, search algorithms).
  * Transparent document reference viewer to inspect complete retrieved text chunks without truncation.
* **Thinking Mode Support**: Pre-configured system prompt with `<|think|>` support for compatible models like `gemma4:26b`.

---

## 📋 Prerequisites

1. **Python**: Python 3.9 or higher.
2. **Ollama**: Download and install [Ollama](https://ollama.com/).
3. **Ollama Models**: Pull the required default models before launching:

### Main LLM / Vision model

```bash
ollama pull gemma4:26b
```

### Embedding model eg. for OCR

```bash
ollama pull embeddinggemma
```

### Run

1. First start Ollama
2. run the command below in terminal
3. open the browser at <http://127.0.0.1:7860> to go to the application

```bash
uv run local-rag-chat
```
