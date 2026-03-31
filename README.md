# 📚 Teacher's AI Knowledge Assistant

> **A RAG-powered (Retrieval-Augmented Generation) web application** that helps teachers upload educational documents, ask natural language questions, generate quizzes, and receive accurate, source-cited answers — all running **100% locally** with no paid API keys.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **📤 Document Upload** | Upload PDF, DOCX, or TXT files with automatic parsing and indexing |
| **💬 Question Answering** | Ask natural language questions and get grounded, cited answers |
| **📝 Quiz Generation** | Auto-create multiple-choice quizzes by topic or document |
| **📄 Summarisation** | Generate concise summaries with compression statistics |
| **📊 Analytics** | Retrieval quality metrics, similarity charts, timing breakdowns |
| **🔍 Source Citations** | Every answer includes the source document and page number |
| **💾 Persistent Storage** | ChromaDB stores your knowledge base across sessions |

---

## 🏗️ Architecture

```
teacher_rag/
├── app.py                        # Streamlit entry point
├── configs/
│   └── config.yaml               # All settings and model parameters
├── src/
│   ├── ingestion/
│   │   ├── parser.py             # PDF / DOCX / TXT extraction
│   │   └── cleaner.py            # Text cleaning & normalisation
│   ├── chunking/
│   │   └── chunker.py            # Fixed-size & recursive splitting
│   ├── embedding/
│   │   └── embedder.py           # Sentence-transformer embeddings
│   ├── vectorstore/
│   │   └── store.py              # ChromaDB vector store wrapper
│   ├── retrieval/
│   │   └── retriever.py          # Semantic search + cross-encoder reranking
│   ├── generation/
│   │   ├── prompt_builder.py     # Prompt templates (QA, summary, quiz)
│   │   └── generator.py          # HuggingFace LLM wrapper
│   ├── pipeline/
│   │   └── rag_pipeline.py       # Unified orchestrator
│   ├── features/
│   │   ├── summarizer.py         # Map-reduce document summarisation
│   │   └── quiz_generator.py     # Multiple-choice quiz generation
│   ├── evaluation/
│   │   └── metrics.py            # Recall@K, faithfulness, relevance
│   └── utils/
│       ├── logger.py             # Centralised logging
│       └── visualizer.py         # Plotly chart helpers
├── vectorstore_data/             # ChromaDB persistent storage (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚙️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **UI Framework** | Streamlit |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB (persistent, local) |
| **LLM** | HuggingFace (`google/flan-t5-large`) |
| **PDF Parsing** | pdfplumber |
| **DOCX Parsing** | python-docx |
| **Chunking** | langchain-text-splitters |
| **Re-ranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Visualisation** | Plotly |

---

## 📦 Requirements

- **Python 3.10+**
- **pip** (package manager)
- ~3 GB disk space for models (downloaded on first run)
- No GPU required (runs on CPU)

---

## 🚀 Setup & Installation

### 1. Clone / navigate to the project

```bash
cd teacher_rag
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Getting Started (Quick Demo)

1. Open the app and navigate to the **🏠 Home** page.
2. Click **"📦 Load Demo Content"** to load three built-in educational texts (Biology, History, Mathematics).
3. Go to **💬 Ask Questions** and try:
   - *"What is photosynthesis?"*
   - *"When did World War II begin?"*
   - *"Who invented calculus?"*

### Uploading Your Own Documents

1. Navigate to **📤 Document Management**.
2. Click the file uploader and select PDF, DOCX, or TXT files.
3. Click **"Upload & Index"** — the progress bar shows each step.
4. Your documents are now searchable in the knowledge base.

### Asking Questions

1. Go to **💬 Ask Questions**.
2. Type your question in the chat input.
3. The assistant will search your documents and provide a sourced answer.
4. Expand **"📎 Sources"** to see exactly which documents and pages were used.
5. Adjust retrieval settings (Top-K, re-ranking, threshold) in the right sidebar.

### Generating Quizzes

1. Navigate to **📝 Quiz Generator**.
2. Choose to generate by **Topic** or **Document**.
3. Set the number of questions and difficulty level.
4. Click **"Generate Quiz"** and interact with the questions.
5. Download the quiz as a text file for classroom use.

### Summarising Documents

1. Go to **📄 Document Summarizer**.
2. Select a document from the dropdown.
3. Click **"Generate Summary"** to get a concise overview.
4. View compression statistics and download the summary.

---

## 🔧 Configuration

All settings are in `configs/config.yaml`. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 512 | Characters per chunk |
| `chunk_overlap` | 64 | Overlap between chunks |
| `chunking_strategy` | recursive | `fixed` or `recursive` |
| `embedding_model` | all-MiniLM-L6-v2 | Sentence transformer model |
| `top_k` | 5 | Number of chunks retrieved |
| `rerank` | true | Enable cross-encoder re-ranking |
| `generation_model` | google/flan-t5-large | Primary LLM |
| `max_new_tokens` | 512 | Max generation length |
| `similarity_threshold` | 0.3 | Minimum retrieval score |

---

## 🧪 How RAG Works

1. **Ingestion:** Documents are parsed, cleaned, and split into overlapping chunks.
2. **Embedding:** Each chunk is encoded into a 384-dimensional vector using a sentence transformer.
3. **Storage:** Vectors and metadata are stored in ChromaDB for persistent, fast retrieval.
4. **Retrieval:** When a question is asked, it is embedded and the most similar chunks are found via cosine similarity.
5. **Re-ranking:** A cross-encoder model re-scores the top candidates for higher precision.
6. **Generation:** The retrieved context is injected into a prompt and passed to the LLM, which generates a grounded answer.

---

## 📊 Evaluation Metrics

The **Analytics & Evaluation** page provides:

- **Answer Relevance:** Cosine similarity between question and answer embeddings.
- **Context Faithfulness:** Word-overlap score between the answer and retrieved context.
- **Retrieval Scores:** Per-chunk similarity and re-rank scores visualised as charts.
- **Timing Breakdown:** Retrieval vs generation time in pie and waterfall charts.

---

## 🛡️ Privacy & Security

- **100% local processing** — no data leaves your machine.
- **No API keys required** — all models run locally via HuggingFace.
- **Persistent storage** — your knowledge base survives app restarts.
- **No telemetry** — zero tracking or data collection.

---

## 📝 License

This project is for educational purposes.

---

## 🤝 Acknowledgements

- [Streamlit](https://streamlit.io/) — UI framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — LLM inference
- [Sentence-Transformers](https://www.sbert.net/) — Text embeddings
- [ChromaDB](https://www.trychroma.com/) — Vector store
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF extraction
