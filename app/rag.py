"""
RAG (Retrieval-Augmented Generation) core logic for NotesGPT.
Handles document loading, chunking, indexing, and querying.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document



# ==================== Configuration Constants ====================
# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Query parameters
TOP_K = 5

# Model configuration
CHAT_MODEL_NAME = "llama-3.1-8b-instant"  # Free Groq model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Free HuggingFace embeddings

# Vector store configuration
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "notesgpt"


# ==================== Model Initialization ====================
def get_embeddings():
    """Initialize and return HuggingFace embeddings model (free, local)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_llm():
    """Initialize and return ChatGroq model (free API)."""
    return ChatGroq(model=CHAT_MODEL_NAME, temperature=0)


def get_vector_store():
    """Get or create the Chroma vector store."""
    embeddings = get_embeddings()
    
    # Ensure directory exists
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    return vector_store


# ==================== Document Loading ====================
def load_file_to_documents(path: str) -> List[Document]:
    """
    Load a file and convert it to LangChain Document objects.
    
    Args:
        path: Path to the file to load
        
    Returns:
        List of Document objects
    """
    file_extension = Path(path).suffix.lower()
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(path)
    else:
        # Default to text loader for .txt and other text files
        loader = TextLoader(path, encoding='utf-8')
    
    documents = loader.load()
    return documents


# ==================== Document Chunking ====================
def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        docs: List of Document objects to chunk
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs


# ==================== Indexing ====================
def index_document(file_path: str, display_name: str) -> int:
    """
    Index a document into the vector store.
    
    Args:
        file_path: Path to the file to index
        display_name: Display name for the document (usually filename)
        
    Returns:
        Number of chunks added to the vector store
    """
    # Load documents
    docs = load_file_to_documents(file_path)
    
    # Add metadata to each document
    for doc in docs:
        doc.metadata["source"] = display_name
        # Preserve page information if it exists (from PyPDFLoader)
        if "page" not in doc.metadata and "page_number" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page_number"]
    
    # Chunk documents
    chunked_docs = chunk_documents(docs)
    
    # Get vector store and add documents
    vector_store = get_vector_store()
    vector_store.add_documents(chunked_docs)
    
    return len(chunked_docs)


# ==================== Querying ====================
def query_notes(question: str, k: int = TOP_K) -> Dict[str, Any]:
    """
    Query the indexed notes using RAG.
    
    Args:
        question: The question to answer
        k: Number of chunks to retrieve
        
    Returns:
        Dictionary containing:
        - answer: The generated answer
        - citations: List of source citations with snippets
    """
    # Get vector store and create retriever
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # If no documents found, return early
    if not retrieved_docs:
        return {
            "answer": "I couldn't find anything related to that in your notes. Please make sure you've uploaded relevant documents.",
            "citations": []
        }
    
    # Build context for the LLM
    context_parts = []
    citations = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        
        # Build source header
        source_header = f"[{i}] Source: {source}"
        if page is not None:
            source_header += f" (Page {page + 1})"  # Page numbers are 0-indexed
        
        context_parts.append(f"{source_header}\n{doc.page_content}\n")
        
        # Prepare citation for response
        snippet = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
        citation = {
            "id": i,
            "source": source,
            "snippet": snippet
        }
        if page is not None:
            citation["page"] = page + 1
        
        citations.append(citation)
    
    # Build the prompt
    context = "\n".join(context_parts)
    
    prompt = f"""You are a helpful study assistant. Answer the question based ONLY on the context provided below. 

Context from the user's notes:
{context}

Instructions:
- Use ONLY the information from the context above to answer the question.
- If you cannot find the answer in the context, clearly state that you don't know or that the information is not in the provided notes.
- Cite your sources by using the citation numbers [1], [2], etc. when referencing specific information.
- Be clear, concise, and accurate.

Question: {question}

Answer:"""
    
    # Get LLM and generate answer
    llm = get_llm()
    response = llm.invoke(prompt)
    
    # Extract the text content from the response
    answer_text = response.content if hasattr(response, 'content') else str(response)
    
    return {
        "answer": answer_text,
        "citations": citations
    }
