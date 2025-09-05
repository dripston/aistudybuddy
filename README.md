# AI Study Buddy

An AI-powered study assistant that helps you understand your documents through intelligent Q&A. Upload PDFs, ask questions, and get contextual answers with references.

## Features

- 📄 **PDF Processing**: Upload and process PDF documents with OCR fallback
- 🤖 **AI Chat**: Ask questions about your documents and get intelligent responses
- 🔍 **Smart Search**: Vector-based semantic search through your documents
- 📚 **References**: Get source references for every answer
- 💬 **Chat History**: Maintain conversation context
- 🎨 **Modern UI**: Clean, responsive Streamlit interface

## Architecture

- **Backend**: FastAPI with RAG (Retrieval-Augmented Generation) pipeline
- **Frontend**: Streamlit web application
- **Vector Store**: Pinecone for document embeddings
- **LLM**: Sambanova API (Meta-Llama-3.3-70B-Instruct)
- **Embeddings**: E5-Mistral-7B-Instruct model

## Prerequisites

1. **Python 3.8+**
2. **API Keys**:
   - Sambanova API key for LLM and embeddings
   - Pinecone API key for vector storage
3. **System Dependencies**:
   - Tesseract OCR (for PDF text extraction)
   - Poppler (for PDF to image conversion)

### Installing System Dependencies

#### Windows
```bash
# Install Tesseract
choco install tesseract

# Install Poppler
choco install poppler
```

#### macOS
```bash
# Install Tesseract
brew install tesseract

# Install Poppler
brew install poppler
```

#### Ubuntu/Debian
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr

# Install Poppler
sudo apt-get install poppler-utils
```

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Navigate to your project directory
cd aistudybuddy

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Configure Environment Variables

Create a `.env` file in the backend directory:

```bash
# Create .env file
touch backend/.env
```

Add the following to `backend/.env`:

```env
# Sambanova API Configuration
LLM_BASE_URL=https://api.sambanova.ai/v1
LLM_API_KEY=your_sambanova_api_key_here
LLM_MODEL=Meta-Llama-3.3-70B-Instruct

# Embedding Configuration
EMBED_BASE_URL=https://api.sambanova.ai/v1
EMBED_API_KEY=your_sambanova_api_key_here
EMBED_MODEL=E5-Mistral-7B-Instruct

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=ai-study
```

### 3. Setup Pinecone Index

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new index with:
   - **Name**: `ai-study`
   - **Dimension**: `4096` (for E5-Mistral-7B-Instruct embeddings)
   - **Metric**: `cosine`

### 4. Install Dependencies

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
pip install -r requirements.txt
```

## Running the Application

### Option 1: Run Both Services Separately

#### Terminal 1 - Backend (FastAPI)
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2 - Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py --server.port 8501
```

### Option 2: Run with Scripts (Recommended)

Create these helper scripts in your project root:

#### `start_backend.bat` (Windows) / `start_backend.sh` (macOS/Linux)

**Windows (`start_backend.bat`):**
```batch
@echo off
cd backend
call ..\venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
```

**macOS/Linux (`start_backend.sh`):**
```bash
#!/bin/bash
cd backend
source ../venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### `start_frontend.bat` (Windows) / `start_frontend.sh` (macOS/Linux)

**Windows (`start_frontend.bat`):**
```batch
@echo off
cd frontend
call ..\venv\Scripts\activate
streamlit run app.py --server.port 8501
pause
```

**macOS/Linux (`start_frontend.sh`):**
```bash
#!/bin/bash
cd frontend
source ../venv/bin/activate
streamlit run app.py --server.port 8501
```

Make scripts executable on macOS/Linux:
```bash
chmod +x start_backend.sh start_frontend.sh
```

## Usage

1. **Start the Backend**: Run the FastAPI server (it will be available at http://localhost:8000)
2. **Start the Frontend**: Run the Streamlit app (it will open at http://localhost:8501)
3. **Upload Documents**: Use the sidebar to upload PDF files
4. **Ask Questions**: Type questions in the chat interface
5. **View References**: Expand the references section to see source citations

## API Endpoints

- `GET /` - Health check
- `GET /health` - System health status
- `POST /upload-pdf` - Upload and process PDF documents
- `POST /ask` - Ask questions to the RAG system
- `POST /embeddings` - Generate embeddings for text

## Troubleshooting

### Common Issues

1. **Backend Offline Error**:
   - Ensure FastAPI server is running on port 8000
   - Check that all environment variables are set correctly

2. **PDF Upload Fails**:
   - Verify Tesseract and Poppler are installed
   - Check file permissions and disk space

3. **No Embeddings Generated**:
   - Verify Sambanova API key is valid
   - Check internet connection
   - Ensure Pinecone index is configured correctly

4. **Import Errors**:
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

### Logs and Debugging

- Backend logs appear in the terminal running uvicorn
- Frontend logs appear in the Streamlit terminal
- Check browser console for frontend JavaScript errors

## Development

### Project Structure
```
aistudybuddy/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Data models
│   │   ├── services/            # Core services
│   │   │   ├── rag_graph.py     # RAG pipeline
│   │   │   ├── chunk.py         # Document chunking
│   │   │   ├── embeddings.py    # Embedding generation
│   │   │   ├── vectorstore.py   # Pinecone operations
│   │   │   └── pdf.py           # PDF processing
│   │   └── settings.py          # Configuration
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit application
│   └── requirements.txt
└── README.md
```

### Adding New Features

1. **Backend**: Add new routes in `backend/app/routes/`
2. **Frontend**: Modify `frontend/app.py` to add new UI components
3. **Services**: Add new functionality in `backend/app/services/`

## License

This project is open source and available under the MIT License.
