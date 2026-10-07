# Local PDF RAG Chatbot

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot that lets you ask questions about a local PDF document.

Instead of asking the language model to answer from its general knowledge, the application first retrieves relevant sections from the PDF and then provides those sections to the model as context.

---

## How It Works

The application is divided into two main phases.

### Phase 1: Preparation

This happens automatically when the application starts.

**1. Text Extraction**

The app uses **PyMuPDF** to read the PDF page by page and extract its text.

Each extracted section keeps its original page number.

```text
PDF
 ↓
Page 1 → Text
Page 2 → Text
Page 3 → Text
...
```

**2. Chunking**

Long page text is divided into smaller overlapping chunks.

The current configuration is:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
```

Each chunk also stores its page number.

**3. Embeddings**

Each chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

through Sentence Transformers.

**4. FAISS Index**

The embeddings are normalized and stored in a local, in-memory **FAISS** index.

The application uses:

```python
faiss.IndexFlatIP
```

Because the embeddings are normalized, the inner product is equivalent to **cosine similarity**.

At this point, the PDF has been converted into a searchable vector representation.

---

### Phase 2: Live Chat

This happens whenever the user asks a question.

**1. Question Embedding**

The user's question is converted into an embedding using the same:

```text
all-MiniLM-L6-v2
```

model.

**2. Retrieval**

The question embedding is compared with the PDF chunk embeddings using FAISS.

The application retrieves the top 3 most relevant chunks:

```python
TOP_K = 3
```

These chunks contain both their text and original PDF page numbers.

**3. Augmentation**

The retrieved chunks are inserted into a prompt together with the user's question.

The model is instructed to use only the provided PDF context and not invent information.

Conceptually:

```text
User Question
      +
Retrieved PDF Context
      ↓
    Prompt
```

**4. Generation**

The prompt is sent to the Groq API, which runs the language model:

```text
openai/gpt-oss-20b
```

The model generates the final answer using the retrieved PDF context.

**5. Source Pages**

The application displays the PDF pages associated with the retrieved chunks.

Example:

```text
Source: Page 4, Page 7
```

This makes it easier to trace the answer back to the original document.

---

## Tech Stack

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| **Python**                | Application logic               |
| **Streamlit**             | Web interface                   |
| **PyMuPDF**               | PDF text extraction             |
| **Sentence Transformers** | Text embeddings                 |
| **FAISS**                 | Vector similarity search        |
| **Groq API**              | LLM inference                   |
| **python-dotenv**         | Environment variable management |

### Embedding Model

```text
all-MiniLM-L6-v2
```

Runs locally and converts text into semantic vector representations.

### Vector Search

```text
FAISS IndexFlatIP
```

Normalized embeddings are used so that inner product search corresponds to cosine similarity.

### Language Model

```text
openai/gpt-oss-20b
```

Accessed through the Groq API.

---

## Project Structure

```text
pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .env
├── .venv/
└── pdf_name.pdf
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Bilal-Ishtiyaque/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=actual_groq_api_key_here
```

### 5. Add your PDF

Place your PDF in the project directory and configure the filename in `app.py`:

```python
PDF_FILE = "pdf_name"
```

### 6. Start the application

```bash
streamlit run app.py
```

The application should open in your browser.

---

## Configuration

The main RAG parameters are defined in `app.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3
```

### Chunk Size

Controls the approximate size of each text chunk.

Larger chunks provide more context, while smaller chunks can make retrieval more precise.

### Chunk Overlap

Allows neighboring chunks to share some text.

This helps reduce the chance of important information being split between two chunks.

### Top-K

Controls how many relevant chunks are retrieved for each question.

The current application retrieves:

```text
Top 3 chunks
```

---

## Example

A user might ask:

```text
What is the main purpose of the company?
```

The application performs:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Similarity Search
   ↓
Top 3 Relevant Chunks
   ↓
Groq LLM
   ↓
Answer
   ↓
Source Pages
```

The final response might look like:

```text
The company focuses on ...

Source: Page 3, Page 8
```

---

This allows the model to answer questions using information from the specific document.

---

## Limitations

This is a learning-focused RAG implementation rather than a production system.

Current limitations include:

* Supports a single PDF configured in the source code
* Uses character-based chunking
* Does not currently support OCR for scanned PDFs
* Uses an in-memory FAISS index
* Does not persist the vector index to disk
* Retrieves only the top 3 chunks
* Does not use a reranking model
* Does not include automated RAG evaluation
* Does not support authentication or multiple users
* Answer quality depends on PDF extraction, chunking, retrieval, and LLM quality
