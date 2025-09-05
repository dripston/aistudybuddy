# AI Study Buddy - Streamlit Deployment Guide

This is a unified Streamlit application that combines both frontend and backend functionality in a single app. Perfect for deployment on Streamlit Cloud!

## 🚀 Quick Start (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API keys:**
   - Create a `.env` file or use Streamlit secrets
   - Add your Sambanova and Pinecone API keys (see configuration below)

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

That's it! The app will be available at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub
1. Create a new GitHub repository
2. Push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/ai-study-buddy.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set the main file path: `app.py`
5. Click "Deploy"

### Step 3: Configure Secrets
In your Streamlit Cloud app settings, add these secrets:

```toml
# Sambanova API Configuration
LLM_BASE_URL = "https://api.sambanova.ai/v1"
LLM_API_KEY = "your_sambanova_api_key_here"
LLM_MODEL = "Meta-Llama-3.3-70B-Instruct"

# Embedding Configuration  
EMBED_BASE_URL = "https://api.sambanova.ai/v1"
EMBED_API_KEY = "your_sambanova_api_key_here"
EMBED_MODEL = "E5-Mistral-7B-Instruct"
EMBED_DIM = "4096"

# Pinecone Configuration
PINECONE_API_KEY = "your_pinecone_api_key_here"
PINECONE_INDEX = "ai-study"
```

## 🔧 Configuration

### API Keys Required

1. **Sambanova API Key**:
   - Get from [Sambanova Cloud](https://cloud.sambanova.ai/)
   - Used for both LLM and embeddings

2. **Pinecone API Key**:
   - Get from [Pinecone Console](https://app.pinecone.io/)
   - Create an index named "ai-study" with dimension 4096

### Environment Variables

You can configure the app using either:

#### Option A: `.env` file (for local development)
```env
LLM_BASE_URL=https://api.sambanova.ai/v1
LLM_API_KEY=your_sambanova_api_key_here
LLM_MODEL=Meta-Llama-3.3-70B-Instruct
EMBED_BASE_URL=https://api.sambanova.ai/v1
EMBED_API_KEY=your_sambanova_api_key_here
EMBED_MODEL=E5-Mistral-7B-Instruct
EMBED_DIM=4096
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=ai-study
```

#### Option B: Streamlit secrets (for deployment)
Add the same variables to `.streamlit/secrets.toml` or in Streamlit Cloud's secrets management.

## 📁 Project Structure

```
aistudybuddy/
├── app.py                    # Main Streamlit application (unified frontend + backend)
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies (for Streamlit Cloud)
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # Local secrets template
├── .env                     # Local environment variables (create this)
└── README_STREAMLIT_DEPLOYMENT.md
```

## ✨ Features

- 📄 **PDF Upload & Processing**: Upload PDFs with OCR fallback
- 🤖 **AI Chat Interface**: Ask questions about your documents
- 🔍 **Semantic Search**: Vector-based document retrieval
- 📚 **Source References**: See exactly where answers come from
- 💬 **Chat History**: Maintain conversation context
- ⚙️ **Configuration Status**: Real-time API status monitoring
- 🎨 **Modern UI**: Clean, responsive design

## 🛠️ System Dependencies

The app requires these system packages (automatically installed on Streamlit Cloud via `packages.txt`):
- `tesseract-ocr` - For OCR text extraction
- `poppler-utils` - For PDF to image conversion
- `tesseract-ocr-eng` - English language pack for Tesseract

## 🔍 Troubleshooting

### Common Issues

1. **"Configuration Issues" warning**:
   - Check that all API keys are set correctly
   - Verify Pinecone index exists and is accessible

2. **PDF upload fails**:
   - Ensure the PDF is not corrupted
   - Check file size limits (Streamlit Cloud has limits)

3. **No responses from AI**:
   - Verify Sambanova API key is valid
   - Check if you have API credits/quota

4. **Pinecone errors**:
   - Ensure index "ai-study" exists with dimension 4096
   - Check Pinecone API key permissions

### Getting Help

- Check the app's status indicators in the sidebar
- Look at the Streamlit Cloud logs for detailed error messages
- Ensure all environment variables are properly set

## 🎯 Usage

1. **Configure API Keys**: Set up your Sambanova and Pinecone credentials
2. **Upload Documents**: Use the sidebar to upload PDF files
3. **Wait for Processing**: The app will chunk and embed your document
4. **Ask Questions**: Type questions in the chat interface
5. **View References**: Expand reference sections to see sources

## 🚀 Advanced Configuration

### Custom Models
You can change the models by updating these environment variables:
- `LLM_MODEL`: Change the language model
- `EMBED_MODEL`: Change the embedding model
- `EMBED_DIM`: Update if using different embedding dimensions

### Pinecone Settings
- `PINECONE_INDEX`: Change the index name
- The app will automatically create the index if it doesn't exist

## 📝 Notes for Deployment

- The app automatically handles async operations within Streamlit
- All backend functionality is embedded in the single `app.py` file
- System dependencies are automatically installed via `packages.txt`
- Configuration is handled through Streamlit's secrets management
- The app includes comprehensive error handling and status monitoring

Perfect for deployment on Streamlit Cloud with just a few clicks! 🎉
