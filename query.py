#!/usr/bin/env python3
"""
query.py — Conversational Query Interface for Local RAG Engine

Connects to a local LM Studio server running Gemma 4 27B A4B and
retrieves context from a ChromaDB vector database built by ingest.py.

Type your question at the prompt. Type 'quit' or 'exit' to end.
"""

import sys
import os

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ── Configuration ────────────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "gemma-4"          # LM Studio model identifier

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "local_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RETRIEVAL_K = 4                       # number of chunks to retrieve
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions using ONLY \
the provided context from the user's private documents. Follow these rules:

1. Base your answer strictly on the context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite the source document name when possible.
4. Be concise and direct.

Context:
{context}
"""

USER_TEMPLATE = "{question}"


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        # Clean up the source path for display
        source_name = os.path.basename(source)
        formatted.append(
            f"[Source {i}: {source_name}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def build_chain():
    """Assemble the RAG chain: retriever → prompt → LLM → output."""

    # Embedding model (must match the one used in ingest.py)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load the existing vector store
    if not os.path.exists(CHROMA_DIR):
        print(f"[ERROR] Vector database not found at '{CHROMA_DIR}'.")
        print(f"        Run 'python ingest.py' first to build it.")
        sys.exit(1)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()
    if count == 0:
        print("[ERROR] Vector database is empty. Add PDFs to data/ and run ingest.py.")
        sys.exit(1)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    # LLM via LM Studio's OpenAI-compatible endpoint
    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",          # LM Studio doesn't validate keys
        model=LM_STUDIO_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    # LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, count


def main():
    print("=" * 60)
    print("  Local RAG Engine — Document Query Interface")
    print("  Model: Gemma 4 27B A4B via LM Studio")
    print("=" * 60)
    print()

    print("Initializing...")
    try:
        chain, chunk_count = build_chain()
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        sys.exit(1)

    print(f"Ready. {chunk_count} chunks loaded from vector database.")
    print("Type your question below. Type 'quit' or 'exit' to end.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        try:
            print("\nSearching documents...\n")
            answer = chain.invoke(question)
            print(f"Assistant: {answer}\n")
        except Exception as e:
            print(f"[ERROR] {e}\n")
            print("Make sure LM Studio's local server is running.\n")


if __name__ == "__main__":
    main()
