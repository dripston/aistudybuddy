from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

EMBED_DIM = int(os.getenv("EMBED_DIM", 4096))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "aistudybuddy-index")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # change if needed
    )

index = pc.Index(PINECONE_INDEX)

def upsert_embeddings(embeddings, doc_id: str, filename: str):
    """
    Upserts embeddings into Pinecone.

    Args:
        embeddings: list of dicts like
            { "embedding": [...], "chunk_index": int, "text": str }
        doc_id: uuid for the document
        filename: original file name
    """
    vectors = []
    for item in embeddings:
        vectors.append({
            "id": f"{doc_id}_{item['chunk_index']}",
            "values": item["embedding"],
            "metadata": {
                "doc_id": doc_id,
                "source": filename,
                "chunk_index": item["chunk_index"],
                "text": item["text"][:2000]
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)