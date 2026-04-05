# Local RAG Engine — Private Document Intelligence with Gemma 4

A lightweight, high-performance Retrieval-Augmented Generation (RAG) pipeline designed to run entirely offline on macOS. Query a private library of PDF documents using conversational AI — no data ever leaves your machine.

## Overview

Most LLMs are limited to the knowledge they were trained on. This project implements a **RAG architecture**, giving the LLM "temporary memory" by searching through a local vector database of your documents and injecting relevant context into the prompt at query time.

By using **LM Studio** and **Gemma 4 27B A4B** (a Mixture-of-Experts model that activates only ~4B parameters per inference), this engine delivers strong reasoning performance with modest hardware requirements — and ensures **total privacy**. Ideal for analyzing sensitive research, legal documents, or proprietary company data.

## Tech Stack

| Component         | Technology                                      |
| ----------------- | ----------------------------------------------- |
| **LLM Engine**    | LM Studio (serving Gemma 4 27B A4B)             |
| **Orchestration** | LangChain                                       |
| **Vector DB**     | ChromaDB (persistent, local storage)             |
| **Embeddings**    | `all-MiniLM-L6-v2` via HuggingFace (runs locally)|
| **Language**      | Python 3.10+                                     |
| **OS Target**     | macOS (optimized for Apple Silicon)              |

### Hardware Recommendations

| Unified Memory | Experience                                                        |
| -------------- | ----------------------------------------------------------------- |
| 8 GB           | Not recommended — model may fail to load or swap heavily          |
| 16 GB          | Functional with smaller quants (Q4); expect moderate latency      |
| 24 GB+         | Smooth experience; recommended minimum for Gemma 4 27B A4B       |

## Project Structure

```
├── data/              # Place your PDF files here
├── chroma_db/         # Persistent vector database (auto-generated)
├── ingest.py          # Process PDFs and build/update the vector database
├── query.py           # Conversational interface for querying your documents
├── requirements.txt   # Pinned Python dependencies
└── README.md          # This file
```

## Installation & Setup

### 1. Prerequisites

- **Python 3.10+** installed
- **LM Studio** installed — download from [lmstudio.ai](https://lmstudio.ai)
- **Gemma 4 27B A4B** model downloaded within LM Studio

### 2. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/scottski78/Local-RAG-Engine-Private-Document-Intelligence-with-Gemma-4.git
cd Local-RAG-Engine-Private-Document-Intelligence-with-Gemma-4

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1 — Start the Local Inference Server

1. Open **LM Studio**
2. Load the **Gemma 4 27B A4B** model
3. Navigate to the **Local Server** tab (the `<->` icon)
4. Confirm the port is set to **1234**
5. Click **Start Server**

### Step 2 — Add Your Documents

Place all PDFs you want to query into the `data/` folder. Subdirectories are supported — the ingestion script walks the entire tree.

### Step 3 — Run the Ingestion Pipeline

This reads your PDFs, splits them into overlapping chunks, generates embedding vectors, and stores everything in ChromaDB.

```bash
python ingest.py
```

You only need to re-run this when you add, remove, or modify documents in `data/`.

### Step 4 — Start Querying

```bash
python query.py
```

Type a natural-language question at the prompt. The system retrieves the most relevant document chunks, injects them as context, and uses Gemma 4 to generate a grounded answer. Type `quit` or `exit` to end the session.

## How It Works

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Your Query   │────▶│  Embedding Model  │────▶│  ChromaDB    │
│  (natural     │     │  (all-MiniLM-L6)  │     │  Similarity  │
│   language)   │     └───────────────────┘     │  Search      │
└──────────────┘                                └──────┬───────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Answer       │◀────│  Gemma 4 27B A4B  │◀────│  Augmented   │
│  (grounded    │     │  (LM Studio)      │     │  Prompt      │
│   in docs)    │     └───────────────────┘     │  (query +    │
└──────────────┘                                │   context)   │
                                                └──────────────┘
```

1. **Embed** — Your question is converted into a vector using the same embedding model used during ingestion.
2. **Retrieve** — ChromaDB performs a similarity search to find the text chunks most relevant to your question.
3. **Augment** — The retrieved chunks and your question are assembled into a structured prompt.
4. **Generate** — The augmented prompt is sent to Gemma 4 running locally in LM Studio, which produces a response grounded in your documents.

## Configuration

Both scripts define connection settings at the top of the file. The defaults match LM Studio's out-of-the-box configuration:

| Setting              | Default                         | Where            |
| -------------------- | ------------------------------- | ---------------- |
| LM Studio base URL   | `http://localhost:1234/v1`      | `query.py`       |
| Embedding model      | `all-MiniLM-L6-v2`             | both scripts     |
| Chunk size            | 1000 characters                 | `ingest.py`      |
| Chunk overlap         | 200 characters                  | `ingest.py`      |
| Retrieved chunks (k) | 4                               | `query.py`       |
| ChromaDB path         | `./chroma_db`                   | both scripts     |

## Troubleshooting

**Connection refused / timeout** — Confirm LM Studio's local server is running and the port in `query.py` matches (default `1234`).

**Empty or irrelevant answers** — Re-run `python ingest.py` after adding new files. Verify your PDFs contain extractable text (scanned image-only PDFs won't work without OCR).

**Slow responses** — The 27B A4B model benefits from 24 GB+ unified memory. Close other heavy applications. If latency is unacceptable, try a smaller quant or a lighter model.

**Import errors** — Make sure your virtual environment is activated (`source venv/bin/activate`) and dependencies are installed (`pip install -r requirements.txt`).

## Alternative Embedding Option

If you'd prefer to keep everything within a single local inference stack, you can swap `all-MiniLM-L6-v2` for **`nomic-embed-text`** served via [Ollama](https://ollama.com). This eliminates the HuggingFace `sentence-transformers` dependency entirely. See the [LangChain Ollama embeddings docs](https://python.langchain.com/docs/integrations/text_embedding/ollama/) for integration details.

## License

MIT
