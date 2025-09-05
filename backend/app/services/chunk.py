# backend/app/services/chunk.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import uuid

def chunk_document(doc_text: str, doc_id: str = None, 
                   chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Dict]:
    """
    Splits a document into chunks with overlap, keeps track of doc_id + chunk_index.
    
    Args:
        doc_text (str): Full extracted text of the document.
        doc_id (str, optional): Unique ID for the document (auto-generated if not provided).
        chunk_size (int, optional): Size of each chunk.
        chunk_overlap (int, optional): Overlap between chunks.

    Returns:
        List[Dict]: List of chunks with metadata.
    """

    if not doc_id:
        doc_id = str(uuid.uuid4())  # generate unique ID if not provided

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = splitter.split_text(doc_text)

    chunked_docs = []
    for idx, chunk in enumerate(chunks):
        chunked_docs.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk
        })

    return chunked_docs
