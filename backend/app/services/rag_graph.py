# backend/app/services/rag_graph.py

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import httpx
import asyncio
from pinecone import Pinecone

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

# Config
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.sambanova.ai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "Meta-Llama-3.3-70B-Instruct")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.sambanova.ai/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "E5-Mistral-7B-Instruct")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ai-study")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

async def get_embedding(text: str):
    """
    Gets embedding for text using Sambanova API
    """
    if not text or not text.strip():
        return None

    url = f"{EMBED_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": EMBED_MODEL, "input": text}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            data = resp.json()

            if not data or "data" not in data or len(data["data"]) == 0:
                return None

            item = data["data"][0]
            embedding = item.get("embedding") or item.get("vector") or item.get("values")

            if embedding is None:
                return None

            # Convert all values to float for Pinecone compatibility
            embedding = [float(value) for value in embedding]
            return embedding

    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

async def query_llm(prompt: str, max_tokens: int = 1000):
    """
    Queries the LLM with a prompt
    """
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful AI study assistant. Use the provided context to answer the user's question. If you don't know the answer, say so."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            data = resp.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                print(f"Unexpected LLM response: {data}")
                return "I'm sorry, I couldn't generate a response at this time."
                
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "I'm sorry, I encountered an error while processing your request."

class RAGGraph:
    def __init__(self):
        self.state = {
            "question": None,
            "history": [],
            "filters": {},
            "top_k": 5,
            "retrieved": [],
            "context": "",
            "answer": "",
            "citations": []
        }
    
    async def retrieve(self, question: str, filters: Dict = None, top_k: int = 5):
        """
        Retrieve relevant documents from Pinecone
        """
        # Get embedding for the question
        query_embedding = await get_embedding(question)
        if query_embedding is None:
            return []
        
        # Query Pinecone
        try:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            retrieved_docs = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                retrieved_docs.append({
                    "id": match.get('id'),
                    "score": match.get('score'),
                    "text": metadata.get('text', ''),
                    "source": metadata.get('source', ''),
                    "chunk_index": metadata.get('chunk_index', 0)
                })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict]):
        """
        Format retrieved documents into context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Document {i+1} (From: {doc['source']}, Score: {doc['score']:.3f}):\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def generate_prompt(self, question: str, context: str, history: List[Dict] = None):
        """
        Generate prompt for LLM
        """
        history_text = ""
        if history:
            for turn in history[-3:]:  # Use last 3 turns for context
                history_text += f"User: {turn.get('question', '')}\nAssistant: {turn.get('answer', '')}\n\n"
        
        prompt = f"""Based on the following context information, answer the user's question. 
If the context doesn't contain the answer, say you don't know. Don't make up information.

Previous conversation:
{history_text}

Context information:
{context}

User question: {question}

Answer:"""
        
        return prompt
    
    async def __call__(self, question: str, history: List[Dict] = None, filters: Dict = None, top_k: int = 5):
        """
        Execute the RAG pipeline
        """
        # Update state
        self.state["question"] = question
        self.state["history"] = history or []
        self.state["filters"] = filters or {}
        self.state["top_k"] = top_k
        
        # Retrieve relevant documents
        self.state["retrieved"] = await self.retrieve(question, filters, top_k)
        
        # Format context
        self.state["context"] = self.format_context(self.state["retrieved"])
        
        # Generate prompt
        prompt = self.generate_prompt(question, self.state["context"], history)
        
        # Query LLM
        self.state["answer"] = await query_llm(prompt)
        
        # Extract citations
        self.state["citations"] = [
            {"source": doc["source"], "chunk_index": doc["chunk_index"]} 
            for doc in self.state["retrieved"]
        ]
        
        return self.state

# Create a global instance
rag_graph = RAGGraph()