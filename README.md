# Lexi 🚀  
**A Smart, Chrono-Powered Storage Sync & Vector Engine for RAG Applications**

Lexi is a **world-class storage synchronization and semantic indexing engine** designed for modern **Retrieval-Augmented Generation (RAG)** pipelines.  
It combines **Chrono** (file change tracking) with **Lex** (semantic chunking + vector persistence) to keep your knowledge base **always fresh, searchable, and scalable**.

---

## ✨ Key Features

- 🔄 **Incremental File Synchronization**
  - Tracks **new, modified, unchanged, and deleted files**
  - Zero reprocessing of unchanged data

- 🧠 **Semantic Chunking & Embeddings**
  - Converts files into intelligent semantic chunks
  - Embeds content using **Ollama Embeddings**

- 📦 **Persistent Vector Storage**
  - Stores embeddings efficiently in **FAISS**
  - Enables fast similarity search for RAG

- 🗂️ **Deletion-Aware Indexing**
  - Automatically removes vectors when files are deleted
  - Keeps the vector database perfectly in sync with disk

- ⚙️ **Production-Ready Architecture**
  - Thread-safe
  - SQLite-backed metadata store
  - Designed for large directory trees


---



