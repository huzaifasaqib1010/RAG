import os
import hashlib
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 700       # tokenizer tokens
CHUNK_OVERLAP = 100    # tokenizer tokens
TOP_K = 5


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the open-source embedding model once per app instance."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    """Create Groq client using Streamlit Secrets or environment variable."""
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(uploaded_file) -> List[Dict]:
    """
    Extract text page-by-page so that retrieved chunks can point
    back to their original PDF page.
    """
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Token-based chunking
# -----------------------------
def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Tokenize text using the embedding model's tokenizer and create
    overlapping chunks.

    This is preferable to blindly splitting by characters because
    the chunk size is based on actual model tokens.
    """
    if not text.strip():
        return []

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(token_ids) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))

        chunk = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=True,
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(token_ids):
            break

        start += step

    return chunks


def create_chunks(pages: List[Dict], tokenizer) -> List[Dict]:
    """Create token-based chunks while preserving page metadata."""
    chunks = []

    for page in pages:
        page_chunks = chunk_text_by_tokens(
            page["text"],
            tokenizer,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                    "chunk": chunk_number,
                }
            )

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_faiss_index(
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
):
    """
    Create normalized embeddings and an inner-product FAISS index.
    With normalized vectors, inner product approximates cosine similarity.
    """
    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(
    query: str,
    index,
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Tuple[Dict, float]]:
    """Retrieve the most similar chunks for a user question."""
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((chunks[int(idx)], float(score)))

    return results


# -----------------------------
# RAG prompt / generation
# -----------------------------
def build_context(retrieved_chunks: List[Tuple[Dict, float]]) -> str:
    context_parts = []

    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[SOURCE {i} | PDF page {chunk['page']} | "
            f"similarity {score:.3f}]\n{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: List[Tuple[Dict, float]],
    client: Groq,
) -> str:
    context = build_context(retrieved_chunks)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied PDF context.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the answer is not present in the context, say:
   "I couldn't find that information in the uploaded document."
3. Cite the relevant PDF page number(s) in your answer using [Page X].
4. Give a clear, direct answer.
5. If the question asks for an explanation, explain it using the document context.
"""

    user_prompt = f"""
PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF, build a FAISS vector index, and ask questions "
    "using retrieved document context."
)

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=TOP_K,
        step=1,
    )

    st.markdown("### Architecture")
    st.write("PDF → Text → Token chunks → Embeddings → FAISS → Groq GPT-OSS")

    st.markdown("### Models")
    st.write(f"Embedding: `{EMBEDDING_MODEL}`")
    st.write(f"LLM: `{GROQ_MODEL}`")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    # Hash the uploaded file so the app does not rebuild the index
    # unnecessarily on Streamlit reruns.
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("Reading and indexing your PDF..."):
            # Re-create a file-like object because getvalue() was used above.
            import io

            pdf_file = io.BytesIO(file_bytes)
            pdf_file.name = uploaded_file.name

            pages = extract_pdf_text(pdf_file)

            if not pages:
                st.error(
                    "No selectable text was found in this PDF. "
                    "This version works with text-based PDFs. "
                    "Scanned PDFs require OCR."
                )
                st.stop()

            embedding_model = load_embedding_model()

            chunks = create_chunks(
                pages,
                embedding_model.tokenizer,
            )

            if not chunks:
                st.error("No text chunks could be created from this PDF.")
                st.stop()

            index = build_faiss_index(
                chunks,
                embedding_model,
            )

            st.session_state["file_hash"] = file_hash
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

        st.success(
            f"Indexed **{len(pages)} pages** into **{len(chunks)} chunks**."
        )

    else:
        pages = st.session_state["pages"]
        chunks = st.session_state["chunks"]
        index = st.session_state["index"]

        st.info(
            f"Loaded: **{st.session_state['file_name']}** · "
            f"{len(pages)} pages · {len(chunks)} chunks"
        )

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        try:
            embedding_model = load_embedding_model()
            groq_client = load_groq_client()

            with st.spinner("Searching the document..."):
                retrieved_chunks = retrieve(
                    question,
                    index,
                    chunks,
                    embedding_model,
                    top_k,
                )

            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    groq_client,
                )

            st.chat_message("user").write(question)

            with st.chat_message("assistant"):
                st.write(answer)

                with st.expander("🔎 Retrieved sources"):
                    for i, (chunk, score) in enumerate(
                        retrieved_chunks,
                        start=1,
                    ):
                        st.markdown(
                            f"**Source {i} — Page {chunk['page']} "
                            f"— Similarity: {score:.3f}**"
                        )
                        st.write(chunk["text"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")

else:
    st.info("👆 Upload a PDF to begin.")

    st.markdown(
        """
### How this RAG app works

1. **PDF extraction** — `pypdf` extracts text page-by-page.
2. **Tokenization** — the embedding model tokenizer converts text into tokens.
3. **Chunking** — overlapping token chunks are created.
4. **Embedding** — Sentence-Transformers converts each chunk into a vector.
5. **Vector database** — FAISS stores the vectors locally in memory.
6. **Retrieval** — the question is embedded and the most similar chunks are found.
7. **Generation** — retrieved chunks are sent to Groq's GPT-OSS model.
8. **Grounded answer** — the model is instructed to answer only from the PDF and cite pages.
        """
    )
