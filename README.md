# Document Q&A RAG Chatbot with Message History

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload **TXT** or **PDF** documents, ask questions, and receive answers grounded in the uploaded document content. The chatbot also maintains **conversation history** to support follow-up and clarification questions.

## Features

- Upload **TXT** and **PDF** files
- Load documents using LangChain loaders
- Split text into chunks using `RecursiveCharacterTextSplitter`
- Generate embeddings using **HuggingFace Embeddings**
- Store embeddings in **ChromaDB**
- Retrieve relevant document chunks for answering questions
- Use **chat history** for context-aware follow-up questions
- Limit conversation history by token size
- Streamlit chat interface for interactive Q&A
- Grounded answers with fallback: **"I don't know."**

## Tech Stack

- **Streamlit** – user interface
- **LangChain** – RAG pipeline and prompt handling
- **HuggingFace Embeddings** – free embeddings
- **ChromaDB** – vector database
- **Groq** – LLM for answer generation
- **PyPDF** – PDF loading support

## Project Workflow

1. User uploads a TXT or PDF file
2. The document is loaded using LangChain loaders
3. Text is split into chunks using `RecursiveCharacterTextSplitter`
4. Chunks are converted into embeddings using HuggingFace
5. Embeddings are stored in ChromaDB
6. A retriever fetches relevant chunks for each user query
7. Retrieved context + chat history + current question are passed to the LLM
8. The chatbot generates an answer grounded only in the retrieved context

## Prompt Behavior

The chatbot is instructed to:

- answer **only from retrieved context**
- **not use outside knowledge**
- reply with **"I don't know."** if the answer is not found in the retrieved context

