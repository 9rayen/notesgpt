/**
 * NotesGPT Frontend JavaScript
 * Handles file uploads, chat interactions, and library management
 */

// ==================== DOM Elements ====================
const uploadForm = document.getElementById('upload-form');
const filesInput = document.getElementById('files');
const fileNames = document.getElementById('file-names');
const uploadStatus = document.getElementById('upload-status');

const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const answerContainer = document.getElementById('answer-container');
const answerDiv = document.getElementById('answer');
const sourcesDiv = document.getElementById('sources');
const sourcesCount = document.getElementById('sources-count');

const totalDocuments = document.getElementById('total-documents');
const totalChunks = document.getElementById('total-chunks');
const libraryList = document.getElementById('library-list');
const refreshLibrary = document.getElementById('refresh-library');


// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    loadLibrary();
});


// ==================== Library Management ====================
async function loadLibrary() {
    try {
        const response = await fetch('/api/library');
        const data = await response.json();
        
        // Update stats
        totalDocuments.textContent = data.total_documents || 0;
        totalChunks.textContent = data.total_chunks || 0;
        
        // Display documents
        if (data.documents && data.documents.length > 0) {
            libraryList.innerHTML = '';
            data.documents.forEach(doc => {
                const docDiv = document.createElement('div');
                docDiv.className = 'library-item';
                
                const typeIcon = getDocumentIcon(doc.type);
                
                docDiv.innerHTML = `
                    <div class="doc-info">
                        <span class="doc-icon">${typeIcon}</span>
                        <span class="doc-name">${escapeHtml(doc.name)}</span>
                        <span class="doc-chunks">${doc.chunks} chunks</span>
                    </div>
                    <button class="btn-delete" onclick="deleteDocument('${escapeHtml(doc.name)}')">🗑️ Delete</button>
                `;
                
                libraryList.appendChild(docDiv);
            });
        } else {
            libraryList.innerHTML = '<p class="empty-library">📭 Your library is empty. Upload some documents to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading library:', error);
        libraryList.innerHTML = '<p class="error">Error loading library</p>';
    }
}

function getDocumentIcon(type) {
    const icons = {
        'pdf': '📄',
        'word': '📝',
        'excel': '📊',
        'csv': '📈',
        'powerpoint': '📊',
        'image_ocr': '🖼️',
        'unknown': '📄'
    };
    return icons[type] || '📄';
}

async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}" from your library?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/library/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert(`✅ ${data.message}`);
            loadLibrary(); // Refresh the library
        } else {
            alert(`❌ ${data.message}`);
        }
    } catch (error) {
        alert(`❌ Error deleting document: ${error.message}`);
    }
}

refreshLibrary.addEventListener('click', () => {
    loadLibrary();
});


// ==================== File Input Display ====================
filesInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length === 0) {
        fileNames.textContent = 'No files selected';
    } else if (files.length === 1) {
        fileNames.textContent = files[0].name;
    } else {
        fileNames.textContent = `${files.length} files selected`;
    }
});


// ==================== Upload Handler ====================
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const files = filesInput.files;
    if (files.length === 0) {
        showUploadStatus('Please select at least one file', 'error');
        return;
    }
    
    // Show loading message
    showUploadStatus('📥 Uploading & indexing documents...', 'loading');
    
    // Build FormData
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        if (data.status === 'ok') {
            let message = `✅ Successfully indexed ${data.successful}/${data.total_files} files:\n`;
            data.uploaded.forEach(item => {
                if (item.error) {
                    message += `\n❌ ${item.filename}: Error - ${item.error}`;
                } else {
                    message += `\n✓ ${item.filename}: ${item.chunks} chunks`;
                }
            });
            showUploadStatus(message, 'success');
            
            // Reset form
            uploadForm.reset();
            fileNames.textContent = 'No files selected';
            
            // Refresh library
            loadLibrary();
        }
    } catch (error) {
        showUploadStatus(`❌ Error: ${error.message}`, 'error');
    }
});


// ==================== Chat Handler ====================
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question) {
        return;
    }
    
    // Show loading state
    answerContainer.style.display = 'block';
    answerDiv.innerHTML = '<div class="loading">🔍 Searching your library...</div>';
    sourcesDiv.innerHTML = '';
    sourcesCount.textContent = '';
    
    // Build FormData
    const formData = new FormData();
    formData.append('question', question);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display answer
        answerDiv.innerHTML = `<p>${formatAnswer(data.answer)}</p>`;
        
        // Show search info
        if (data.sources_searched) {
            sourcesCount.textContent = `📚 Searched ${data.sources_searched} document(s) and found ${data.total_chunks || data.citations.length} relevant chunks`;
        }
        
        // Display citations
        if (data.citations && data.citations.length > 0) {
            sourcesDiv.innerHTML = '';
            data.citations.forEach(citation => {
                const citationDiv = document.createElement('div');
                citationDiv.className = 'citation';
                
                const typeIcon = getDocumentIcon(citation.type);
                
                let citationHtml = `
                    <div class="citation-header">
                        <strong>[${citation.id}]</strong> ${typeIcon} ${escapeHtml(citation.source)}
                `;
                
                if (citation.page !== undefined) {
                    citationHtml += ` <span class="page-number">(Page ${citation.page})</span>`;
                } else if (citation.slide !== undefined) {
                    citationHtml += ` <span class="page-number">(Slide ${citation.slide})</span>`;
                } else if (citation.sheet !== undefined) {
                    citationHtml += ` <span class="page-number">(Sheet: ${escapeHtml(citation.sheet)})</span>`;
                }
                
                citationHtml += `
                    </div>
                    <div class="citation-snippet">${escapeHtml(citation.snippet)}</div>
                `;
                
                citationDiv.innerHTML = citationHtml;
                sourcesDiv.appendChild(citationDiv);
            });
        } else {
            sourcesDiv.innerHTML = '<p class="no-sources">No sources available</p>';
        }
        
    } catch (error) {
        answerDiv.innerHTML = `<p class="error">❌ Error: ${escapeHtml(error.message)}</p>`;
        sourcesDiv.innerHTML = '';
    }
});


// ==================== Helper Functions ====================
function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
    uploadStatus.style.display = 'block';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    // Convert line breaks to <br> and preserve formatting
    return escapeHtml(text).replace(/\n/g, '<br>');
}
