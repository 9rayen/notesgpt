"""
FastAPI application for NotesGPT.
Handles file uploads, indexing, and chat queries.
"""

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from app.rag import index_document, query_notes


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="NotesGPT", description="RAG-based study assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Ensure uploads directory exists
UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Routes ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_notes(files: List[UploadFile] = File(...)):
    """
    Upload and index one or more document files.
    
    Args:
        files: List of files to upload and index
        
    Returns:
        JSON response with upload status and chunk counts
    """
    uploaded = []
    
    for file in files:
        # Save file to uploads directory
        file_path = UPLOADS_DIR / file.filename
        
        # Write file content
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Index the document
        try:
            num_chunks = index_document(str(file_path), file.filename)
            uploaded.append({
                "filename": file.filename,
                "chunks": num_chunks
            })
        except Exception as e:
            uploaded.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "status": "ok",
        "uploaded": uploaded
    }


@app.post("/api/chat")
async def chat(question: str = Form(...)):
    """
    Answer a question using the RAG pipeline.
    
    Args:
        question: The question to answer
        
    Returns:
        JSON response with answer and citations
    """
    result = query_notes(question)
    return result


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 NotesGPT is starting up...")
    print(f"📁 Uploads directory: {UPLOADS_DIR.absolute()}")
    print(f"🗄️  Vector store: ./chroma_db")
    print("✅ Ready to serve at http://127.0.0.1:8000")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
