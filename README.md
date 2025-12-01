# NotesGPT 📚

**NotesGPT** is an AI-powered study assistant that uses Retrieval-Augmented Generation (RAG) to help you interact with your study notes intelligently. Upload your PDF or text documents, and ask questions to get accurate answers with source citations.

## 🎯 Features

- **Document Upload & Indexing**: Upload PDF and plain text files
- **Intelligent Chunking**: Documents are split into optimal chunks for better retrieval
- **Vector Search**: Uses ChromaDB for efficient semantic search
- **RAG Pipeline**: Retrieves relevant context before generating answers
- **Source Citations**: Every answer includes references to source documents with page numbers
- **Persistent Storage**: Your indexed documents persist between sessions
- **Simple UI**: Clean, user-friendly web interface

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **AI/RAG Framework**: LangChain
- **Vector Database**: ChromaDB (persistent storage)
- **LLM**: Groq (llama-3.1-8b-instant - FREE API)
- **Embeddings**: HuggingFace (sentence-transformers - FREE, runs locally)
- **Document Processing**: pypdf, TextLoader
- **Frontend**: HTML, CSS, Vanilla JavaScript

## 📋 Prerequisites

- Python 3.10 or higher
- **Free Groq API key** - Get it at: https://console.groq.com (no credit card required!)

## 🚀 Quick Start

### 1. Clone or Download

Download this project to your local machine.

### 2. Create Virtual Environment

```powershell
cd notesgpt
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root (or edit the existing one):

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Important**: 
1. Go to https://console.groq.com and sign up (FREE, no credit card needed)
2. Get your API key from the dashboard
3. Replace `your_groq_api_key_here` with your actual Groq API key

### 5. Run the Application

```powershell
uvicorn app.main:app --reload
```

Or simply:

```powershell
python -m app.main
```

### 6. Open in Browser

Navigate to: **http://127.0.0.1:8000**

## 📖 Usage

### Uploading Notes

1. Click the **"Choose files..."** button in the Upload section
2. Select one or more PDF or TXT files
3. Click **"Upload & Index"**
4. Wait for the system to process and index your documents
5. You'll see a confirmation with the number of chunks created

### Asking Questions

1. Type your question in the text area in the Chat section
2. Click **"Ask NotesGPT"**
3. Wait for the AI to retrieve relevant context and generate an answer
4. View the answer along with source citations below

### Example Questions

- "What are the key concepts in chapter 3?"
- "Summarize the main arguments about [topic]"
- "What does the document say about [specific term]?"

## 🧠 How RAG Works

**RAG (Retrieval-Augmented Generation)** is a powerful technique that combines:

1. **Retrieval**: Searches your indexed documents for the most relevant chunks
2. **Augmentation**: Provides those chunks as context to the language model
3. **Generation**: The LLM generates an answer based solely on the provided context

This approach ensures:
- ✅ Answers are grounded in your actual documents
- ✅ Reduced hallucinations (making up information)
- ✅ Transparent sourcing with citations
- ✅ Up-to-date information from your notes

## ⚙️ Configuration

You can customize the RAG pipeline by editing constants in `app/rag.py`:

```python
# Chunking parameters
CHUNK_SIZE = 800          # Size of each text chunk
CHUNK_OVERLAP = 150       # Overlap between chunks

# Query parameters
TOP_K = 5                 # Number of chunks to retrieve

# Model configuration
CHAT_MODEL_NAME = "llama-3.1-8b-instant"  # Free Groq model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Free local embeddings
```

## 📁 Project Structure

```
notesgpt/
├── app/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # FastAPI app & routes
│   └── rag.py            # RAG logic (indexing & querying)
├── templates/
│   └── index.html        # Main UI template
├── static/
│   ├── app.js            # Frontend JavaScript
│   └── styles.css        # CSS styling
├── uploads/              # Uploaded files (created at runtime)
├── chroma_db/            # ChromaDB persistent storage (created at runtime)
├── .env                  # Environment variables (not in git)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure
- The `uploads/` and `chroma_db/` directories contain your data

## 🐛 Troubleshooting

### "No Groq API key found"
Make sure your `.env` file exists and contains `GROQ_API_KEY=your_key_here`

### "Module not found" errors
Run `pip install -r requirements.txt` to install all dependencies

### ChromaDB errors
Delete the `chroma_db/` folder and re-index your documents

### Port already in use
Change the port in the run command: `uvicorn app.main:app --port 8001`

## 🚀 Future Enhancements

- Support for more document formats (DOCX, Markdown, etc.)
- Multiple collection support
- Chat history
- Advanced filtering and metadata search
- User authentication
- Document management (delete, update)

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Groq](https://groq.com/) - FREE Lightning-fast AI API
- [HuggingFace](https://huggingface.co/) - FREE local embeddings

---

**Happy studying with NotesGPT! 📚✨**
