# 🚀 ContextAI – Context-Aware Chat for Webpages & Documents 🤖

Ever wished you had a super-smart assistant that could **read a webpage or document and answer questions about it instantly**?

**ContextAI** is a Retrieval-Augmented Generation (RAG) system that lets you:

- Ingest **webpages or uploaded files**
- Convert their contents into **vector embeddings**
- Ask questions through a **context-aware chatbot**

Unlike traditional AI chatbots, ContextAI **only answers using the information you provide**, ensuring responses remain **grounded in the source material**.

---

# ✨ Key Features

### 🌐 Webpage Ingestion
Provide any public URL and the system will automatically:

- scrape readable content
- chunk the text
- generate embeddings
- store vectors in a database for retrieval

---

### 📂 Multi-Format File Support
Upload documents directly into the RAG pipeline.

Supported formats:

- **PDF**
- **DOCX**
- **TXT**
- **Excel (.xlsx / .xls)**
- **PowerPoint (.pptx)**
- **Images (.png / .jpg / .jpeg)** via OCR

This makes the system useful for **document analysis, research, and knowledge extraction**.

---

### 🧠 Context-Grounded Chat
The chatbot answers questions **only using the ingested content**.

This prevents hallucinations and ensures responses remain **source-accurate**.

---

### 🔎 Semantic Search with Vector Database
Each text chunk is converted into embeddings and stored in **Qdrant Cloud**.

When a user asks a question:

1. The query is embedded
2. Similar chunks are retrieved using vector similarity
3. Retrieved context is passed to Gemini for answer generation

---

### ⚙️ Debug Mode for Retrieval Inspection
A built-in debug mode allows developers to **inspect retrieved chunks** and verify the retrieval pipeline.

---

# 🛠 Tech Stack

| Category | Technology | Role |
|--------|--------|--------|
| **Frontend / UI** | Streamlit | Interactive ingestion and chat interface |
| **Backend** | Python | Core application logic |
| **LLM Provider** | Google Gemini (`gemini-2.5-flash`) | Answer generation |
| **Embeddings** | Gemini Embeddings | Semantic vector generation |
| **Vector Database** | Qdrant Cloud | Vector similarity search |
| **Scraping** | `r.jina.ai` | Clean webpage text extraction |
| **Document Parsing** | PyMuPDF, python-docx, pandas, python-pptx | Extract text from documents |
| **OCR** | Tesseract + Pillow | Extract text from images |

---

# 🧠 How the RAG Pipeline Works

The system follows a **two-stage pipeline**.

## 1️⃣ Ingestion Phase (Knowledge Creation)

When a URL or file is provided:

1. Content is extracted from the source  
2. Text is divided into smaller chunks  
3. Each chunk is converted into embeddings  
4. Embeddings are stored in **Qdrant vector database**

Metadata such as **source name and chunk ID** are stored alongside vectors.

---

## 2️⃣ Retrieval + Generation Phase (Answering Questions)

When a user asks a question:

1. The query is converted into an embedding  
2. Qdrant retrieves the most relevant chunks  
3. Retrieved chunks form the **context**  
4. Context + query are sent to Gemini  
5. Gemini generates the final answer  

The model is instructed to answer **strictly from the retrieved context**.

---

# 🏃 Running the Project Locally

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ContextAI.git
cd ContextAI
```

---

## 2️⃣ Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root.

```
GEMINI_API_KEY=your_gemini_api_key

QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_collection
```

You can obtain:

- Gemini API key from **Google AI Studio**
- Qdrant credentials from **Qdrant Cloud**

---

## 5️⃣ Start the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 6️⃣ Open the App

Navigate to:

```
http://localhost:8501
```

From here you can:

- ingest URLs
- upload documents
- chat with the ingested content

---

# 🔍 Example Use Cases

ContextAI can be used for:

- Research assistants
- Legal document exploration
- Academic paper analysis
- Website knowledge extraction
- Internal company knowledge bases

---

# ⚠️ Notes on Persistence

Currently the system stores vectors in **Qdrant**, while the active context is managed within the application session.

For production deployments, consider adding:

- Redis
- PostgreSQL
- persistent context tracking

to ensure session continuity across restarts.

---

# 📌 Future Improvements

Planned enhancements include:

- multi-document context support
- hybrid search (BM25 + vector)
- conversation memory
- document source citations
- improved chunking strategies

---

# ⭐ Contributing

Pull requests and ideas are welcome!

If you find this project useful, consider giving it a ⭐ on GitHub.

---

# 📜 License

MIT License.
