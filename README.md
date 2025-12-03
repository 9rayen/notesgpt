# NotesGPT 📚

**NotesGPT** is an AI-powered study assistant that uses Retrieval-Augmented Generation (RAG) to help you interact with your study notes intelligently. Upload your PDF or text documents, and ask questions to get accurate answers with source citations.

## 🎯 Features

- **Document Upload & Indexing**: Upload PDF, Word, Excel, PowerPoint, images, text, and markdown files
- **🔍 OCR (Optical Character Recognition)**: Extract text from images and scanned documents
  - Multi-language support (English, French, Arabic, and more)
  - AI-powered text extraction with EasyOCR
  - Image preprocessing for improved accuracy
  - Edit extracted text before indexing
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
- **OCR Engine**: EasyOCR (multi-language, deep learning-based)
- **Image Processing**: OpenCV, Pillow
- **Document Processing**: pypdf, python-docx, python-pptx, openpyxl, pandas, pdf2image
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

### Uploading Documents

1. Click the **"Choose files..."** button in the Upload section
2. Select one or more files (PDF, Word, Excel, PowerPoint, images, text, markdown)
3. Click **"Upload & Index"**
4. Wait for the system to process and index your documents
5. You'll see a confirmation with the number of chunks created

### 🔍 OCR - Extracting Text from Images

**NotesGPT now includes powerful OCR capabilities!** You can scan documents, photos, screenshots, and handwritten notes to extract text.

#### How to Use OCR:

1. **Upload an Image**: 
   - Click **"📷 Choose image..."** in the OCR section
   - Select an image file (PNG, JPG, TIFF, BMP, etc.)
   - Supported formats: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.gif`, `.webp`

2. **Extract Text**:
   - Click **"🔍 Extract Text (OCR)"**
   - The system will process the image (this may take 10-30 seconds on first run)
   - You'll see:
     - Image preview
     - Extracted text
     - Confidence score
     - Word count
     - Detected languages

3. **Review & Edit**:
   - Review the extracted text in the editor
   - Make any corrections if needed
   - The text is fully editable

4. **Add to Library**:
   - Click **"✅ Add to Library"**
   - The extracted text will be indexed and added to your knowledge base
   - You can now ask questions about the content

#### OCR Features:

- **Multi-language Support**: English, French, Arabic, Spanish, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, and more
- **Image Preprocessing**: Automatic noise reduction, contrast enhancement, and adaptive thresholding for better accuracy
- **Handwriting Support**: Can handle both printed text and handwritten text (with varying accuracy)
- **Scanned Documents**: Works great with scanned PDFs converted to images
- **High Accuracy**: Uses EasyOCR, a state-of-the-art deep learning OCR engine

#### OCR Tips for Best Results:

- ✅ Use high-resolution images (300 DPI or higher)
- ✅ Ensure good lighting and contrast
- ✅ Keep text horizontal and properly aligned
- ✅ Avoid shadows, glare, or reflections
- ✅ Use clear, legible fonts for printed text
- ⚠️ Handwritten text works better when neat and clear
- ⚠️ First OCR operation may take longer (downloading language models)

### Asking Questions

1. Type your question in the text area in the Chat section
2. Click **"🔍 Search & Answer"**
3. Wait for the AI to retrieve relevant context and generate an answer
4. View the answer along with source citations below

### Example Questions

- "What are the key concepts in chapter 3?"
- "Summarize the main arguments about [topic]"
- "What does the document say about [specific term]?"
- "Extract the main points from my scanned notes"
- "What information is in the image I uploaded?"

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

### OCR Issues

#### "OCR is taking too long"
- First-time use: EasyOCR downloads language models (~50-100MB), which takes a few minutes
- Large images: Try reducing image size or resolution
- Multiple languages: Using fewer languages can speed up processing

#### "OCR not detecting text"
- Ensure image has good contrast and lighting
- Try preprocessing the image externally (increase contrast, convert to grayscale)
- Check that text is horizontal and not skewed
- Make sure text is large enough (minimum 12pt font recommended)

#### "Low confidence scores"
- Improve image quality (higher resolution, better lighting)
- Remove noise, shadows, or distortions
- Ensure text is in focus
- Try different image formats (PNG usually works best)

#### "Out of memory errors"
- Reduce image size before uploading
- Process one image at a time
- Close other memory-intensive applications
- Consider using GPU acceleration if available

## 🔧 OCR Installation & Setup

The OCR feature requires additional dependencies. After running `pip install -r requirements.txt`, the following packages will be installed:

- **EasyOCR**: Deep learning-based OCR engine
- **OpenCV**: Image preprocessing
- **pdf2image**: PDF to image conversion (for scanned PDFs)

### First-Time Setup:

When you first use OCR, EasyOCR will download language models (~50-100 MB per language). This happens automatically and only needs to be done once.

### System Requirements:

- **Python**: 3.10 or higher
- **RAM**: At least 4GB recommended for OCR
- **Disk Space**: ~500MB for language models
- **Optional**: CUDA-enabled GPU for faster OCR (CPU works fine but slower)

### Enabling GPU Acceleration (Optional):

If you have an NVIDIA GPU with CUDA support, you can enable GPU acceleration for faster OCR:

1. Install CUDA toolkit from NVIDIA
2. Edit `app/ocr.py` and change `gpu=False` to `gpu=True` in the `get_ocr_reader()` function

## 🚀 Future Enhancements

- Batch OCR processing for multiple images
- PDF OCR for scanned documents
- Multiple collection support
- Chat history
- Advanced filtering and metadata search
- User authentication
- Real-time OCR preview

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
