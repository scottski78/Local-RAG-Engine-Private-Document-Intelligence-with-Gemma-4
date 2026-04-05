# Local-RAG-Engine-Private-Document-Intelligence-with-Gemma-4
A lightweight, high-performance Retrieval-Augmented Generation (RAG) pipeline designed to run entirely offline on macOS. This system allows users to perform conversational AI queries against a private library of PDF documents without any data ever leaving their local machine.

🚀 Overview
Most LLMs are limited to the knowledge they were trained on. This project implements a RAG architecture, which provides the LLM with "temporary memory" by searching through a local vector database of your specific documents and injecting relevant snippets into the prompt.

By using LM Studio and Gemma-4, this engine ensures total privacy, making it ideal for analyzing sensitive research, legal documents, or private company data.

🛠️ Tech Stack
LLM Engine: LM Studio (Running Gemma-4)
Orchestration: LangChain
Vector Database: ChromaDB
Embedding Model: all-MiniLM-L6-v2 (Running locally via HuggingFace)
Language: Python 3.10+
OS Target: macOS (Optimized for Apple Silicon)
📂 Project Structure
├── data/               # Place your PDF files here
├── chroma_db/          # The persistent vector database (auto-generated)
├── ingest.py           # Script to process PDFs and update the database
├── query.py            # The user interface for chatting with documents
└── README.md           # Project documentation

⚙️ Installation & Setup
1. Prerequisites
Install LM Studio.
Download the Gemma-4 model within LM Studio.

2. Clone and Environment Setup
# Clone this repository
git clone https://github.com/scottski78/Local-RAG-Engine-Private-Document-Intelligence-with-Gemma-4.git

cd Local-RAG-Engine-Private-Document-Intelligence-with-Gemma-4

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install langchain langchain-openai langchain-community chromadb sentence-transformers pypdf
📖 Usage Guide

Step 1: Start the Local Inference Server
Open LM Studio.
Load the Gemma-4 model.
Navigate to the Local Server tab (<-> icon).
Ensure the port is set to 1234.
Click Start Server.

Step 2: Prepare Your Knowledge Base
Place all the PDF documents you wish to "teach" the AI into the data/ folder.

Step 3: Run the Ingestion Pipeline
This script reads your PDFs, chunks them into manageable pieces, converts them into mathematical vectors, and stores them in ChromaDB.

python ingest.py
Note: You only need to run this script when you add new documents or modify existing ones.

Step 4: Start the Conversation
Launch the query engine to begin asking questions about your data.

python query.py
Once the interface loads, type your question. The system will automatically retrieve the relevant context from your PDFs and use Gemma-4 to generate a precise answer.

🔍 How it Works (The Pipeline)
Retrieval: When you ask a question, query.py uses the embedding model to turn your question into a vector.
Search: It performs a similarity search in ChromaDB to find the text chunks most mathematically similar to your question.
Augmentation: The system "stuffs" those retrieved text chunks into a prompt template along with your original question.
Generation: The augmented prompt is sent to the LM Studio local server, where Gemma-4 generates a natural language response based only on the provided context.

⚠️ Troubleshooting
Connection Error: Ensure LM Studio's local server is active and the URL in query.py matches your LM Studio port (default http://localhost:1234/v1).
No Results Found: Ensure you have run python ingest.py after adding files to the data/ folder.
Slow Response: Large PDFs or high-parameter models require significant RAM/Unified Memory. For best results on Mac, ensure no other heavy applications are running.
