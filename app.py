import streamlit as st
import os
import asyncio
import uuid
from typing import List, Dict, Optional
from io import BytesIO
import time
import hashlib

# Core libraries
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import httpx
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Text processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Please install langchain: pip install langchain")
    st.stop()

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate unique session ID for user isolation
def get_session_id():
    """Generate a unique session ID for user isolation"""
    if "session_id" not in st.session_state:
        # Create a unique session ID based on session info
        session_info = f"{st.session_state.get('_session_id', '')}{time.time()}"
        st.session_state.session_id = hashlib.md5(session_info.encode()).hexdigest()[:16]
    return st.session_state.session_id

# Configuration from environment variables or Streamlit secrets
def get_config(key: str, default: str = None):
    """Get configuration from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    # Fall back to environment variables
    return os.getenv(key, default)

LLM_BASE_URL = get_config("LLM_BASE_URL", "https://api.sambanova.ai/v1")
LLM_API_KEY = get_config("LLM_API_KEY")
LLM_MODEL = get_config("LLM_MODEL", "Meta-Llama-3.3-70B-Instruct")
EMBED_BASE_URL = get_config("EMBED_BASE_URL", "https://api.sambanova.ai/v1")
EMBED_API_KEY = get_config("EMBED_API_KEY")
EMBED_MODEL = get_config("EMBED_MODEL", "E5-Mistral-7B-Instruct")
EMBED_DIM = int(get_config("EMBED_DIM", "4096"))
PINECONE_API_KEY = get_config("PINECONE_API_KEY")
PINECONE_INDEX = get_config("PINECONE_INDEX", "ai-study")

# Initialize Pinecone (with error handling)
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    try:
        if not PINECONE_API_KEY:
            return None, "Pinecone API key not found in environment variables"
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists, create if not
        existing_indexes = [i.name for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            time.sleep(10)
        
        index = pc.Index(PINECONE_INDEX)
        return index, None
    except Exception as e:
        return None, f"Pinecone initialization error: {str(e)}"

# Backend Functions
async def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for text using Sambanova API"""
    if not text or not text.strip():
        return None
    
    if not EMBED_API_KEY:
        st.error("Embedding API key not found in environment variables")
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
        st.error(f"Error getting embedding: {e}")
        return None

def chunk_document(doc_text: str, doc_id: str = None, 
                   chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Dict]:
    """Split document into chunks"""
    if not doc_id:
        doc_id = str(uuid.uuid4())

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

def upsert_embeddings(index, embeddings: List[Dict], doc_id: str, filename: str, session_id: str):
    """Upsert embeddings into Pinecone with session isolation"""
    if not index:
        return False
    
    vectors = []
    for item in embeddings:
        vectors.append({
            "id": f"{session_id}_{doc_id}_{item['chunk_index']}",  # Include session_id for isolation
            "values": item["embedding"],
            "metadata": {
                "session_id": session_id,  # Add session_id to metadata for filtering
                "doc_id": doc_id,
                "source": filename,
                "chunk_index": item["chunk_index"],
                "text": item["text"][:2000]  # Limit text size
            }
        })

    try:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
        return True
    except Exception as e:
        st.error(f"Error upserting to Pinecone: {e}")
        return False

async def query_llm(prompt: str, max_tokens: int = 1000) -> str:
    """Query the LLM with a prompt"""
    if not LLM_API_KEY:
        return "LLM API key not found in environment variables"
    
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
                return "I'm sorry, I couldn't generate a response at this time."
                
    except Exception as e:
        return f"I'm sorry, I encountered an error: {str(e)}"

async def retrieve_documents(index, question: str, session_id: str, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant documents from Pinecone for current user only"""
    if not index:
        return []
    
    # Get embedding for the question
    query_embedding = await get_embedding(question)
    if query_embedding is None:
        return []
    
    try:
        # Filter by session_id to only get current user's documents
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"session_id": session_id}  # Only query current user's documents
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
        st.error(f"Error querying Pinecone: {e}")
        return []

def format_context(retrieved_docs: List[Dict]) -> str:
    """Format retrieved documents into context string"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"Document {i+1} (From: {doc['source']}, Score: {doc['score']:.3f}):\n{doc['text']}\n")
    
    return "\n".join(context_parts)

def generate_prompt(question: str, context: str, history: List[Dict] = None) -> str:
    """Generate prompt for LLM"""
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

async def process_pdf(file_bytes: bytes, filename: str, index, session_id: str) -> Dict:
    """Process PDF file and extract text"""
    try:
        # Extract text from PDF
        reader = PdfReader(BytesIO(file_bytes))
        pages_text = []

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            # OCR fallback if text is empty
            if not text.strip():
                try:
                    images = convert_from_bytes(file_bytes, first_page=i + 1, last_page=i + 1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        text = ocr_text.strip()
                except Exception:
                    text = "[OCR failed]"

            pages_text.append({"page": i + 1, "text": text})

        # Combine all text and chunk
        full_text = "\n".join([p["text"] for p in pages_text])
        chunks = chunk_document(full_text)

        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            embedding = await get_embedding(chunk.get("text", ""))
            if embedding is None:
                continue
            embeddings.append({
                "embedding": embedding,
                "chunk_index": chunk.get("chunk_index", -1),
                "text": chunk.get("text", "")
            })

        # Store in Pinecone with session isolation
        if embeddings and index:
            doc_id = chunks[0].get("doc_id", "unknown") if chunks else "unknown"
            success = upsert_embeddings(index, embeddings, doc_id, filename, session_id)
            pinecone_status = f"upserted {len(embeddings)} vectors" if success else "upsert failed"
        else:
            pinecone_status = "no valid vectors to upsert"

        return {
            "filename": filename,
            "pages": len(pages_text),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "pinecone_status": pinecone_status
        }

    except Exception as e:
        raise Exception(f"PDF processing error: {str(e)}")

async def ask_question(index, question: str, session_id: str, history: List[Dict] = None, top_k: int = 5) -> Dict:
    """Ask a question using RAG pipeline"""
    try:
        # Retrieve relevant documents for current user only
        retrieved = await retrieve_documents(index, question, session_id, top_k)
        
        # Format context
        context = format_context(retrieved)
        
        # Generate prompt
        prompt = generate_prompt(question, context, history)
        
        # Query LLM
        answer = await query_llm(prompt)
        
        # Format references
        references = []
        for doc in retrieved:
            references.append({
                "doc_id": doc.get("id") or doc.get("doc_id", "unknown"),
                "source": doc.get("source", "unknown"),
                "chunk_index": doc.get("chunk_index", -1),
                "score": doc.get("score", 0.0),
                "preview": doc.get("text", "")[:200]
            })

        return {
            "answer": answer,
            "references": references
        }

    except Exception as e:
        return {
            "answer": f"I'm sorry, I encountered an error: {str(e)}",
            "references": []
        }

# Streamlit UI
def main():
    # Get session ID for user isolation
    session_id = get_session_id()
    
    # Modern Dark Theme CSS
    st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #16213e 0%, #0f0f23 100%);
        border-right: 1px solid #333;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Chat Container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background: rgba(26, 26, 46, 0.3);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* User Message */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1.5rem 0;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Assistant Message */
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 1.5rem 0;
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .assistant-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 1.2rem;
        flex-shrink: 0;
        margin-top: 5px;
    }
    
    /* Reference Cards */
    .references-section {
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(15, 15, 35, 0.5);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .reference-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    
    .reference-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }
    
    .reference-title {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .reference-meta {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-bottom: 0.5rem;
    }
    
    .reference-preview {
        font-style: italic;
        color: #d0d0d0;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Status Indicators */
    .status-indicator {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem 0;
    }
    
    .status-online {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.3);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }
    
    /* Configuration Warning */
    .config-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .config-warning a {
        color: #ffffff;
        text-decoration: underline;
        font-weight: 600;
    }
    
    /* Session Info */
    .session-info {
        background: rgba(26, 26, 46, 0.6);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    
    .session-id {
        font-family: 'Courier New', monospace;
        background: rgba(102, 126, 234, 0.2);
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Upload Section */
    .upload-section {
        background: rgba(26, 26, 46, 0.4);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px dashed rgba(102, 126, 234, 0.3);
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(26, 26, 46, 0.6);
    }
    
    /* Welcome Message */
    .welcome-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .welcome-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-steps {
        text-align: left;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .welcome-steps li {
        margin: 0.8rem 0;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 15, 35, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: rgba(26, 26, 46, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Sidebar text color */
    .css-1d391kg .stMarkdown {
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(26, 26, 46, 0.4) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🤖 AI Study Buddy</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your study materials and ask questions to get AI-powered insights!</div>', unsafe_allow_html=True)

    # Initialize Pinecone
    index, pinecone_error = init_pinecone()
    
    # Check configuration
    config_issues = []
    if not LLM_API_KEY:
        config_issues.append("LLM_API_KEY not set")
    if not EMBED_API_KEY:
        config_issues.append("EMBED_API_KEY not set")
    if not PINECONE_API_KEY:
        config_issues.append("PINECONE_API_KEY not set")
    
    if config_issues or pinecone_error:
        st.markdown(f"""
        <div class="config-warning">
            <strong>⚠️ Configuration Issues:</strong><br>
            {"<br>".join(config_issues)}
            {f"<br>{pinecone_error}" if pinecone_error else ""}
            <br><br>
            <strong>To fix this:</strong><br>
            1. Edit <code>.streamlit/secrets.toml</code> and add your API keys<br>
            2. Get Sambanova API key from: <a href="https://cloud.sambanova.ai/" target="_blank">cloud.sambanova.ai</a><br>
            3. Get Pinecone API key from: <a href="https://app.pinecone.io/" target="_blank">app.pinecone.io</a><br>
            4. Create a Pinecone index named "ai-study" with dimension 4096
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Configuration status
        if not config_issues and not pinecone_error:
            st.markdown('<span class="status-indicator status-online">🟢 Configuration OK</span>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-offline">🔴 Configuration Issues</span>', 
                       unsafe_allow_html=True)
        
        # Session info
        st.markdown(f"""
        <div class="session-info">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Session ID</div>
            <div class="session-id">{session_id}</div>
            <div style="font-size: 0.85rem; color: #a0a0a0; margin-top: 0.5rem;">
                🔒 Your documents are private to this session
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        # Upload section
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your study materials in PDF format"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Process PDF", type="primary"):
                if not index:
                    st.error("Cannot process PDF: Pinecone not configured properly")
                else:
                    with st.spinner("Processing PDF... This may take a moment."):
                        try:
                            file_bytes = uploaded_file.read()
                            result = asyncio.run(process_pdf(file_bytes, uploaded_file.name, index, session_id))
                            st.success(f"✅ Successfully processed {result['filename']}")
                            st.info(f"📄 Pages: {result['pages']}")
                            st.info(f"📝 Chunks: {result['chunks']}")
                            st.info(f"🔗 Embeddings: {result['embeddings']}")
                            st.info(f"💾 Status: {result['pinecone_status']}")
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")

        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Number of references",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of document chunks to retrieve for context"
        )
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-bubble">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show references if available
            if "references" in message and message["references"]:
                st.markdown(f"""
                <div class="references-section">
                    <div style="font-weight: 600; margin-bottom: 1rem; color: #667eea;">
                        📚 References ({len(message['references'])} sources)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                for i, ref in enumerate(message["references"], 1):
                    st.markdown(f"""
                    <div class="reference-card">
                        <div class="reference-title">Reference {i}: {ref['source']}</div>
                        <div class="reference-meta">Chunk {ref['chunk_index']} • Relevance Score: {ref.get('score', 0.0):.3f}</div>
                        <div class="reference-preview">"{ref['preview']}..."</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents...", disabled=not index):
        if not index:
            st.error("Please configure your API keys to use the chat functionality.")
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">
                {prompt}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = asyncio.run(ask_question(index, prompt, session_id, st.session_state.chat_history, top_k))
            
            answer = response["answer"]
            references = response.get("references", [])
            
            # Add assistant message to chat history
            assistant_message = {
                "role": "assistant", 
                "content": answer,
                "references": references
            }
            st.session_state.messages.append(assistant_message)
            
            # Update chat history for context
            st.session_state.chat_history.append({
                "question": prompt,
                "answer": answer
            })
            
            # Display assistant message
            st.markdown(f"""
            <div class="assistant-message">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-bubble">
                    {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show references
            if references:
                st.markdown(f"""
                <div class="references-section">
                    <div style="font-weight: 600; margin-bottom: 1rem; color: #667eea;">
                        📚 References ({len(references)} sources)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                for i, ref in enumerate(references, 1):
                    st.markdown(f"""
                    <div class="reference-card">
                        <div class="reference-title">Reference {i}: {ref['source']}</div>
                        <div class="reference-meta">Chunk {ref['chunk_index']} • Relevance Score: {ref.get('score', 0.0):.3f}</div>
                        <div class="reference-preview">"{ref['preview']}..."</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.rerun()

    # Instructions for first-time users
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">👋 Welcome to AI Study Buddy!</div>
            <div class="welcome-steps">
                <strong>To get started:</strong>
                <ol>
                    <li>Configure your API keys (see sidebar for status)</li>
                    <li>Upload a PDF document using the sidebar</li>
                    <li>Wait for it to be processed</li>
                    <li>Ask questions about your document in the chat below</li>
                </ol>
                <strong>Example questions:</strong>
                <ul>
                    <li>"What are the main topics covered in this document?"</li>
                    <li>"Can you summarize chapter 3?"</li>
                    <li>"What does the author say about [specific topic]?"</li>
                </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close chat container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
