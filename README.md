# AI Study Buddy

A modern AI-powered study assistant with a beautiful chat interface. Upload PDFs and get intelligent answers with source citations.

## 🚀 Quick Deploy to Streamlit Cloud

1. **Fork/Clone this repo**
2. **Deploy on [share.streamlit.io](https://share.streamlit.io)**
3. **Add your API keys in Streamlit Cloud secrets**
4. **Done!** ✨

## 🔑 Required API Keys

Add these in Streamlit Cloud's secrets management:

```toml
# Sambanova API (get from https://cloud.sambanova.ai/)
LLM_API_KEY = "your_sambanova_api_key"
EMBED_API_KEY = "your_sambanova_api_key"  # Same as LLM key

# Pinecone API (get from https://app.pinecone.io/)
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX = "ai-study"
```

## ✨ Features

- 🎨 **Modern Dark UI** - ChatGPT-style interface
- 📄 **PDF Processing** - Upload and analyze documents
- 🤖 **AI Chat** - Ask questions about your documents  
- 🔍 **Smart Search** - Vector-based semantic search
- 📚 **Source Citations** - See exactly where answers come from
- 🔒 **Privacy** - Each user's documents are isolated
- ⚡ **Fast** - Optimized for Streamlit Cloud

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Add API keys to .streamlit/secrets.toml
# Run the app
streamlit run app.py
```

## 📋 Setup Pinecone Index

1. Go to [app.pinecone.io](https://app.pinecone.io/)
2. Create index: `ai-study`
3. Dimension: `4096`
4. Metric: `cosine`

## 🎯 How to Use

1. **Upload PDF** - Use sidebar to upload study materials
2. **Wait for Processing** - Documents are chunked and embedded
3. **Ask Questions** - Chat with your documents
4. **View Sources** - See references for each answer

Perfect for students, researchers, and anyone who needs to analyze documents! 🎉
