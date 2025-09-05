# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
from typing import List, Dict

from .services.chunk import chunk_document
from .services.embeddings import get_embedding
from .services.vectorstore import upsert_embeddings
from .services.rag_graph import rag_graph

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello from AI Study Buddy!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/embeddings")
async def create_embedding(input: str = Body(..., embed=True)):
    embedding = await get_embedding(input)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    return {
        "object": "embedding",
        "model": "E5-Mistral-7B-Instruct",
        "data": [
            {"object": "embedding", "embedding": embedding, "index": 0}
        ],
        "usage": {"total_tokens": len(input.split())}
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Save temporary file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text per page
        reader = PdfReader(temp_path)
        pages_text = []

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            # OCR fallback if text is empty
            if not text.strip():
                try:
                    images = convert_from_path(temp_path, first_page=i + 1, last_page=i + 1)
                    ocr_text = pytesseract.image_to_string(images[0])
                    text = ocr_text.strip()
                except Exception:
                    text = "[OCR failed]"

            pages_text.append({"page": i + 1, "text": text})

        # Clean up temp file
        os.remove(temp_path)

        # Combine all text and chunk
        full_text = "\n".join([p["text"] for p in pages_text])
        chunks = chunk_document(full_text)

        # Prepare embeddings for Pinecone - FIXED
        embeddings = []
        for chunk in chunks:
            embedding = await get_embedding(chunk.get("text", ""))
            if embedding is None:
                idx = chunk.get("chunk_index", "unknown")
                print(f"⚠️ Skipping chunk {idx} (no embedding)")
                continue
            embeddings.append({
                "embedding": embedding,
                "chunk_index": chunk.get("chunk_index", -1),
                "text": chunk.get("text", "")
            })

        if embeddings:
            # Use the first chunk's doc_id
            doc_id = chunks[0].get("doc_id", "unknown") if chunks else "unknown"
            upsert_embeddings(embeddings, doc_id, file.filename)
            pinecone_status = f"upserted {len(embeddings)} vectors"
        else:
            pinecone_status = "no valid vectors to upsert"

        return {
            "filename": file.filename,
            "pages": len(pages_text),
            "chunks": len(chunks),
            "pinecone_status": pinecone_status
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(
    question: str = Body(..., embed=True),
    history: List[Dict] = Body(default=[]),
    filters: Dict = Body(default={}),
    top_k: int = Body(default=5)
):
    """
    Ask a question to the RAG system.
    Returns: { answer, references: [{doc_id, source, chunk_index, preview}] }
    """
    try:
        state = await rag_graph(question, history, filters, top_k)

        # Format references properly
        references = []
        for doc in state["retrieved"]:
            references.append({
                "doc_id": doc.get("id") or doc.get("doc_id", "unknown"),
                "source": doc.get("source", "unknown"),
                "chunk_index": doc.get("chunk_index", -1),
                "preview": doc.get("text", "")[:200]  # preview first 200 chars
            })

        return {
            "answer": state["answer"],
            "references": references
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
