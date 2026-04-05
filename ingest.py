#!/usr/bin/env python3
"""
ingest.py — PDF Ingestion Pipeline for Local RAG Engine

Reads all PDFs from the data/ directory, splits them into overlapping
text chunks, generates embeddings using all-MiniLM-L6-v2, and stores
everything in a persistent ChromaDB vector database.

Re-run this script whenever you add, remove, or modify documents.
"""

import os
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ────────────────────────────────────────────────────
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "local_rag"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks
# ─────────────────────────────────────────────────────────────────────


def discover_pdfs(data_dir: str) -> list[Path]:
    """Walk the data directory and return all PDF paths."""
    root = Path(data_dir)
    if not root.exists():
        print(f"[ERROR] Data directory '{data_dir}' does not exist.")
        print(f"        Create it and add your PDF files, then re-run.")
        sys.exit(1)

    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        print(f"[WARNING] No PDF files found in '{data_dir}'.")
        print(f"          Add your documents and re-run.")
        sys.exit(0)

    return pdfs


def load_documents(pdf_paths: list[Path]) -> list:
    """Load and concatenate pages from all PDFs."""
    all_docs = []
    for path in pdf_paths:
        print(f"  Loading: {path}")
        try:
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            all_docs.extend(pages)
        except Exception as e:
            print(f"  [SKIP] Could not read {path.name}: {e}")
    return all_docs


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: list, embedding_model: str, persist_dir: str):
    """Generate embeddings and persist to ChromaDB."""
    print(f"\n  Initializing embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Remove existing database to do a clean rebuild
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print(f"  Cleared previous database at {persist_dir}")

    print(f"  Generating embeddings and writing to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return vectorstore


def main():
    print("=" * 60)
    print("  Local RAG Engine — Ingestion Pipeline")
    print("=" * 60)

    # Discover
    print(f"\n[1/4] Scanning '{DATA_DIR}' for PDFs...")
    pdf_paths = discover_pdfs(DATA_DIR)
    print(f"       Found {len(pdf_paths)} PDF(s).\n")

    # Load
    print("[2/4] Loading documents...")
    documents = load_documents(pdf_paths)
    print(f"       Loaded {len(documents)} page(s) total.\n")

    if not documents:
        print("[ERROR] No readable content extracted. Check your PDFs.")
        sys.exit(1)

    # Chunk
    print("[3/4] Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"       Created {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")

    # Embed + Store
    print("[4/4] Building vector database...")
    t0 = time.time()
    build_vectorstore(chunks, EMBEDDING_MODEL, CHROMA_DIR)
    elapsed = time.time() - t0
    print(f"       Done in {elapsed:.1f}s.\n")

    print("=" * 60)
    print("  Ingestion complete!")
    print(f"  Database: {CHROMA_DIR}")
    print(f"  Documents: {len(pdf_paths)} | Pages: {len(documents)} | Chunks: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
