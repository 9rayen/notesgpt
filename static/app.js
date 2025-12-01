/**
 * NotesGPT Frontend JavaScript
 * Handles file uploads and chat interactions
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
            let message = '✅ Successfully indexed:\n';
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
    answerDiv.innerHTML = '<div class="loading">🤔 Thinking...</div>';
    sourcesDiv.innerHTML = '';
    
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
        answerDiv.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;
        
        // Display citations
        if (data.citations && data.citations.length > 0) {
            sourcesDiv.innerHTML = '';
            data.citations.forEach(citation => {
                const citationDiv = document.createElement('div');
                citationDiv.className = 'citation';
                
                let citationHtml = `
                    <div class="citation-header">
                        <strong>[${citation.id}]</strong> ${escapeHtml(citation.source)}
                `;
                
                if (citation.page !== undefined) {
                    citationHtml += ` <span class="page-number">(Page ${citation.page})</span>`;
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
