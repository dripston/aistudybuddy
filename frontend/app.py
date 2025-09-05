import streamlit as st
import requests
import json
from typing import List, Dict
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API base URL
API_BASE_URL = "http://localhost:8000"

def check_backend_health():
    """Check if the backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_pdf(file):
    """Upload PDF file to backend"""
    files = {"file": (file.name, file, "application/pdf")}
    try:
        response = requests.post(f"{API_BASE_URL}/upload-pdf", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def ask_question(question: str, history: List[Dict] = None, top_k: int = 5):
    """Ask a question to the RAG system"""
    payload = {
        "question": question,
        "history": history or [],
        "filters": {},
        "top_k": top_k
    }
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return None

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    .assistant-message {
        background-color: #f0f8f0;
        border-left-color: #28a745;
    }
    .reference-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #6c757d;
        margin: 0.5rem 0;
    }
    .status-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-online {
        background-color: #d4edda;
        color: #155724;
    }
    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🤖 AI Study Buddy</h1>', unsafe_allow_html=True)
    st.markdown("Upload your study materials and ask questions to get AI-powered insights!")

    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Check backend status
        backend_status = check_backend_health()
        if backend_status:
            st.markdown('<span class="status-indicator status-online">🟢 Backend Online</span>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-offline">🔴 Backend Offline</span>', 
                       unsafe_allow_html=True)
            st.error("Backend is not running. Please start the FastAPI server.")
            st.stop()

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
                with st.spinner("Processing PDF... This may take a moment."):
                    result = upload_pdf(uploaded_file)
                    if result:
                        st.success(f"✅ Successfully processed {result['filename']}")
                        st.info(f"📄 Pages: {result['pages']}")
                        st.info(f"📝 Chunks: {result['chunks']}")
                        st.info(f"🔗 Status: {result['pinecone_status']}")

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
    st.header("💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Assistant:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show references if available
            if "references" in message:
                with st.expander(f"📚 References ({len(message['references'])} sources)"):
                    for i, ref in enumerate(message["references"], 1):
                        st.markdown(f"""
                        <div class="reference-card">
                            <strong>Reference {i}:</strong> {ref['source']}<br>
                            <small>Chunk {ref['chunk_index']} • Score: {ref.get('score', 'N/A')}</small><br>
                            <em>{ref['preview']}...</em>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {prompt}
        </div>
        """, unsafe_allow_html=True)

        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = ask_question(prompt, st.session_state.chat_history, top_k)
            
            if response:
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
                <div class="chat-message assistant-message">
                    <strong>AI Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Show references
                if references:
                    with st.expander(f"📚 References ({len(references)} sources)"):
                        for i, ref in enumerate(references, 1):
                            st.markdown(f"""
                            <div class="reference-card">
                                <strong>Reference {i}:</strong> {ref['source']}<br>
                                <small>Chunk {ref['chunk_index']}</small><br>
                                <em>{ref['preview']}...</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.rerun()
            else:
                st.error("Failed to get response from AI assistant")

    # Instructions for first-time users
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to AI Study Buddy!**
        
        To get started:
        1. Upload a PDF document using the sidebar
        2. Wait for it to be processed
        3. Ask questions about your document in the chat below
        
        Example questions:
        - "What are the main topics covered in this document?"
        - "Can you summarize chapter 3?"
        - "What does the author say about [specific topic]?"
        """)

if __name__ == "__main__":
    main()
