import os

import streamlit as st
import pymupdf
import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

PDF_FILE = ""
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3


# -----------------------------
# Streamlit Configuration
# -----------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("RAG Chatbot")
st.caption("Ask questions about the PDF using retrieval-augmented generation.")


# -----------------------------
# Initialize Models / Clients
# -----------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it to your .env file.")
    st.stop()

groq_client = Groq(api_key=groq_api_key)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# Extracting the PDF (load_pdf)
# -----------------------------

@st.cache_data
def load_pdf():
    """
    Extract text from the PDF while preserving page numbers.

    Returns:
        list[dict]: Each item contains:
            {
                "text": "...",
                "page": 1
            }
    """

    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(
            f"PDF file '{PDF_FILE}' was not found."
        )

    doc = pymupdf.open(PDF_FILE)

    if len(doc) == 0:
        doc.close()
        raise ValueError("The PDF contains no pages.")

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number
                }
            )

    doc.close()

    if not pages:
        raise ValueError(
            "No extractable text was found in the PDF. "
            "The PDF may contain scanned images instead of text."
        )

    return pages


# -----------------------------
# Chopping it into blocks (split_text)
# -----------------------------

def split_text(
    pages,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Split page text into overlapping character-based chunks
    while preserving the page number.

    Returns:
        list[dict]: Each item contains:
            {
                "text": "...",
                "page": 1
            }
    """

    chunks = []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    step = chunk_size - chunk_overlap

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            start += step

    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")

    return chunks


# -----------------------------
# Create Vector Database (core mathematical engine of RAG)
# -----------------------------

@st.cache_resource
def create_vector_store():
    """
    Create a FAISS vector database using cosine similarity.

    Cosine similarity is implemented by:
    1. Normalizing embeddings.
    2. Using FAISS IndexFlatIP.

    Inner product between normalized vectors = cosine similarity.
    """

    pages = load_pdf()

    chunks = split_text(pages)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    # Normalize embeddings so inner product becomes cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, chunks


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------

def retrieve_chunks(question, index, chunks):
    """
    Retrieve the most relevant chunks for a question.

    Returns:
        list[dict]: Retrieved chunks with text, page and score.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    question_embedding = question_embedding.astype("float32")

    # Normalize query embedding for cosine similarity.
    faiss.normalize_L2(question_embedding)

    distances, indices = index.search(
        question_embedding,
        min(TOP_K, len(chunks))
    )

    retrieved_chunks = []

    for score, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number != -1:

            chunk = chunks[index_number].copy()

            chunk["score"] = float(score)

            retrieved_chunks.append(chunk)

    return retrieved_chunks


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using only the retrieved PDF context.
    """

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions about a PDF.

Answer the user's question using ONLY the information provided
in the context below.

Rules:
- Use only information from the context.
- Do not invent facts.
- If the context does not contain enough information to answer
  the question, clearly say that the answer cannot be found
  in the provided PDF context.
- Keep the answer concise and directly answer the question.
- Do not mention the retrieval process.
- Do not mention chunk numbers or similarity scores.

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# -----------------------------
# Create Vector Store
# -----------------------------

try:

    index, chunks = create_vector_store()

except Exception as e:

    st.error(f"Failed to process PDF: {e}")

    st.stop()


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources for previous assistant messages.
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            pages = message["sources"]

            st.caption(
                "Source: "
                + ", ".join(
                    f"Page {page}"
                    for page in pages
                )
            )


# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask something about the PDF..."
)


if question:

    # -------------------------
    # Display User Question
    # -------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -------------------------
    # Retrieve Relevant Chunks
    # -------------------------

    retrieved_chunks = retrieve_chunks(
        question,
        index,
        chunks
    )

    if not retrieved_chunks:

        answer = (
            "I couldn't find relevant information "
            "in the PDF."
        )

        source_pages = []

    else:

        source_pages = sorted(
            set(
                chunk["page"]
                for chunk in retrieved_chunks
            )
        )

        # ---------------------
        # Generate Answer
        # ---------------------

        try:

            answer = generate_answer(
                question,
                retrieved_chunks
            )

        except Exception as e:

            st.error(f"Groq error: {e}")

            st.stop()

    # -------------------------
    # Display Assistant Answer
    # -------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        if source_pages:

            st.caption(
                "Source: "
                + ", ".join(
                    f"Page {page}"
                    for page in source_pages
                )
            )

    # -------------------------
    # Save Assistant Message
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_pages
        }
    )
