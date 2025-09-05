import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

# === Load ENV Vars ===
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_DIM = int(os.getenv("EMBED_DIM"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# === 1. Test SambaNova LLM ===
def test_llm():
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model="Meta-Llama-3.3-70B-Instruct",  # SambaNova LLM
        messages=[{"role": "user", "content": "Hello from AI Study Buddy!"}],
    )
    print("LLM Response:", response.choices[0].message.content)


# === 2. Test Embeddings ===
def test_embeddings():
    embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)
    response = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input="AI Study Buddy test embedding"
    )
    embedding = response.data[0].embedding
    print("Embedding length:", len(embedding))
    return embedding


# === 3. Test Pinecone Insert + Query ===
def test_pinecone(embedding):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn’t exist
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,   # Must match embedding length (4096)
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )
        print(f"Created index {PINECONE_INDEX}")

    # Connect to index
    index = pc.Index(PINECONE_INDEX)

    # Upsert a test vector
    index.upsert([("test-id-1", embedding, {"text": "AI Study Buddy test"})])
    print("Inserted into Pinecone!")
    # After upserting
    query_result = index.query(
        vector=embedding,  # use the same vector we inserted
        top_k=1,
        include_metadata=True
    )
    print("Pinecone Query Result:", query_result)

    # Query it back
    result = index.query(vector=embedding, top_k=1, include_metadata=True)
    print("Pinecone Query Result:", result)


if __name__ == "__main__":
    test_llm()
    emb = test_embeddings()
    test_pinecone(emb)
