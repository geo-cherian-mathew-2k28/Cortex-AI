/**
 * ═══════════════════════════════════════════════════════════════════════
 * CORTEX AI – Frontend Application
 * Enterprise Document Intelligence Platform
 * Handles file upload, chat, streaming, and UI interactions
 * ═══════════════════════════════════════════════════════════════════════
 */

const API_BASE = "";

// ── State ─────────────────────────────────────────────────────────────
const state = {
    sessionId: '',
    files: [],
    isStreaming: false,
    totalChunks: 0,
};

// ── DOM References ────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebarToggle'),
    sidebarExpandBtn: $('#sidebarExpandBtn'),
    sidebarCloseMobile: $('#sidebarCloseMobile'),
    mobileMenuBtn: $('#mobileMenuBtn'),
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    uploadProgress: $('#uploadProgress'),
    progressFill: $('#progressFill'),
    progressText: $('#progressText'),
    fileList: $('#fileList'),
    fileCount: $('#fileCount'),
    emptyState: $('#emptyState'),
    clearSessionBtn: $('#clearSessionBtn'),
    exportChatBtn: $('#exportChatBtn'),
    sessionStatus: $('#sessionStatus'),
    fileFilterWrapper: $('#fileFilterWrapper'),
    fileFilter: $('#fileFilter'),
    chunkBadge: $('#chunkBadge'),
    chunkCount: $('#chunkCount'),
    chatContainer: $('#chatContainer'),
    welcomeScreen: $('#welcomeScreen'),
    messages: $('#messages'),
    chatInput: $('#chatInput'),
    charCount: $('#charCount'),
    sendBtn: $('#sendBtn'),
    summaryModal: $('#summaryModal'),
    summaryTitle: $('#summaryTitle'),
    summaryContent: $('#summaryContent'),
    closeSummaryModal: $('#closeSummaryModal'),
};

// ═══════════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    adjustTextareaHeight();
});

function initEventListeners() {
    // Sidebar toggle & expand
    els.sidebarToggle.addEventListener('click', toggleSidebar);
    els.sidebarExpandBtn.addEventListener('click', expandSidebar);
    els.sidebarCloseMobile.addEventListener('click', closeMobileSidebar);
    els.mobileMenuBtn.addEventListener('click', toggleMobileSidebar);

    // Upload
    els.uploadZone.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', handleFileSelect);
    els.uploadZone.addEventListener('dragover', handleDragOver);
    els.uploadZone.addEventListener('dragleave', handleDragLeave);
    els.uploadZone.addEventListener('drop', handleDrop);

    // Chat
    els.chatInput.addEventListener('input', handleInputChange);
    els.chatInput.addEventListener('keydown', handleKeyDown);
    els.sendBtn.addEventListener('click', sendMessage);

    // Session
    els.clearSessionBtn.addEventListener('click', clearSession);
    els.exportChatBtn.addEventListener('click', exportChat);

    // Modal
    els.closeSummaryModal.addEventListener('click', closeSummaryModal);
    els.summaryModal.addEventListener('click', (e) => {
        if (e.target === els.summaryModal) closeSummaryModal();
    });

    // Capability cards
    $$('.cap-card').forEach(card => {
        card.addEventListener('click', () => {
            const query = card.dataset.query;
            if (query) {
                els.chatInput.value = query;
                handleInputChange();
                els.chatInput.focus();
                // Close mobile sidebar if open
                if (els.sidebar.classList.contains('mobile-open')) {
                    closeMobileSidebar();
                }
            }
        });
    });

    // Mobile overlay
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('mobile-overlay')) {
            toggleMobileSidebar();
        }
    });

    // Escape key to close modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (els.summaryModal.style.display === 'flex') closeSummaryModal();
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  SIDEBAR
// ═══════════════════════════════════════════════════════════════════════
function toggleSidebar() {
    els.sidebar.classList.toggle('collapsed');
    updateExpandBtnVisibility();
}

function expandSidebar() {
    els.sidebar.classList.remove('collapsed');
    updateExpandBtnVisibility();
}

function updateExpandBtnVisibility() {
    if (els.sidebar.classList.contains('collapsed')) {
        els.sidebarExpandBtn.classList.add('visible');
    } else {
        els.sidebarExpandBtn.classList.remove('visible');
    }
}

function toggleMobileSidebar() {
    els.sidebar.classList.toggle('mobile-open');
    let overlay = $('.mobile-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'mobile-overlay';
        document.body.appendChild(overlay);
    }
    overlay.classList.toggle('active');
}

function closeMobileSidebar() {
    els.sidebar.classList.remove('mobile-open');
    const overlay = $('.mobile-overlay');
    if (overlay) overlay.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════════════
//  FILE UPLOAD
// ═══════════════════════════════════════════════════════════════════════
function handleDragOver(e) {
    e.preventDefault();
    els.uploadZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    els.uploadZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    els.uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
    els.fileInput.value = '';
}

async function uploadFiles(fileList) {
    const formData = new FormData();
    for (const file of fileList) {
        formData.append('files', file);
    }
    if (state.sessionId) {
        formData.append('session_id', state.sessionId);
    }

    // Show progress
    els.uploadProgress.style.display = 'block';
    els.progressFill.style.width = '10%';
    els.progressText.textContent = 'Uploading files...';
    els.sessionStatus.textContent = 'Processing...';

    try {
        // Animate progress with stage messages
        let progress = 10;
        const stages = [
            { at: 20, text: 'Parsing document...' },
            { at: 40, text: 'Chunking text...' },
            { at: 55, text: 'Loading AI model (first time may take a moment)...' },
            { at: 70, text: 'Generating embeddings...' },
            { at: 80, text: 'Building search index...' },
        ];
        let stageIdx = 0;

        const progressInterval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 5, 88);
            els.progressFill.style.width = `${progress}%`;

            if (stageIdx < stages.length && progress >= stages[stageIdx].at) {
                els.progressText.textContent = stages[stageIdx].text;
                stageIdx++;
            }
        }, 500);

        const response = await fetch(`${API_BASE}/api/upload?session_id=${state.sessionId}`, {
            method: 'POST',
            body: formData,
        });

        clearInterval(progressInterval);
        els.progressFill.style.width = '100%';
        els.progressText.textContent = 'Complete!';

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Upload failed (${response.status})`);
        }

        const data = await response.json();

        // Update session
        state.sessionId = data.session_id;
        state.totalChunks = data.total_chunks;

        // Process results
        let successCount = 0;
        for (const result of data.results) {
            if (result.status === 'success') {
                state.files.push(result);
                successCount++;
            } else {
                showNotification(`Error: ${result.message}`, 'error');
            }
        }

        updateFileList();
        updateFilterSelect();
        updateChunkBadge();
        els.sessionStatus.textContent = 'Ready';

        if (successCount > 0) {
            showNotification(`${successCount} file(s) processed successfully`, 'info');
        }

        // Auto-generate summary for the first file
        if (data.results.length === 1 && data.results[0].status === 'success') {
            setTimeout(() => openSummary(data.results[0].filename), 500);
        }

    } catch (error) {
        console.error('Upload error:', error);
        els.progressText.textContent = 'Upload failed';
        els.sessionStatus.textContent = 'Error';
        showNotification(error.message, 'error');
    } finally {
        setTimeout(() => {
            els.uploadProgress.style.display = 'none';
            els.progressFill.style.width = '0%';
        }, 1500);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  FILE LIST UI
// ═══════════════════════════════════════════════════════════════════════
function updateFileList() {
    if (state.files.length === 0) {
        els.emptyState.style.display = 'block';
        els.fileCount.textContent = '0';
        return;
    }

    els.emptyState.style.display = 'none';
    els.fileCount.textContent = state.files.length;

    // Clear existing cards (but keep empty state)
    $$('.file-card').forEach(c => c.remove());

    state.files.forEach(file => {
        const card = document.createElement('div');
        card.className = 'file-card';
        card.innerHTML = `
            <div class="file-type-badge ${file.file_type}">${file.file_type}</div>
            <div class="file-info">
                <div class="file-name" title="${file.filename}">${file.filename}</div>
                <div class="file-meta">
                    <span>${formatNumber(file.word_count)} words</span>
                    <span>${file.chunk_count} chunks</span>
                    <span>${file.size_mb}MB</span>
                </div>
            </div>
            <button class="file-action-btn" title="View Summary" data-file="${file.filename}">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5"/>
                    <path d="M8 5v3M8 10v1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
            </button>
        `;

        card.addEventListener('click', (e) => {
            if (e.target.closest('.file-action-btn')) {
                openSummary(file.filename);
            }
        });

        els.fileList.appendChild(card);
    });
}

function updateFilterSelect() {
    // Clear existing options except "All"
    while (els.fileFilter.options.length > 1) {
        els.fileFilter.remove(1);
    }

    state.files.forEach(file => {
        const option = document.createElement('option');
        option.value = file.filename;
        option.textContent = file.filename;
        els.fileFilter.appendChild(option);
    });

    els.fileFilterWrapper.style.display = state.files.length > 1 ? 'flex' : 'none';
}

function updateChunkBadge() {
    els.chunkBadge.style.display = state.totalChunks > 0 ? 'flex' : 'none';
    els.chunkCount.textContent = state.totalChunks;
}

// ═══════════════════════════════════════════════════════════════════════
//  CHAT
// ═══════════════════════════════════════════════════════════════════════
function handleInputChange() {
    const val = els.chatInput.value;
    els.charCount.textContent = `${val.length}/2000`;
    els.sendBtn.disabled = val.trim().length === 0 || state.isStreaming;
    adjustTextareaHeight();
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!els.sendBtn.disabled) {
            sendMessage();
        }
    }
}

function adjustTextareaHeight() {
    const ta = els.chatInput;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
}

async function sendMessage() {
    const query = els.chatInput.value.trim();
    if (!query || state.isStreaming) return;

    // Hide welcome, show messages
    if (els.welcomeScreen.style.display !== 'none') {
        els.welcomeScreen.style.display = 'none';
        els.messages.style.display = 'flex';
    }

    // Show export button once conversation starts
    els.exportChatBtn.style.display = 'flex';

    // Add user message
    appendMessage('user', query);

    // Clear input
    els.chatInput.value = '';
    handleInputChange();

    // Start streaming
    state.isStreaming = true;
    els.sendBtn.disabled = true;
    els.sessionStatus.textContent = 'Thinking...';

    // Add assistant message placeholder with typing indicator
    const assistantMsg = appendMessage('assistant', '', true);

    try {
        const fileFilter = els.fileFilter.value || null;

        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                session_id: state.sessionId,
                file_filter: fileFilter,
                stream: true,
            }),
        });

        if (!response.ok) {
            throw new Error(`Chat failed: ${response.statusText}`);
        }

        // Update session ID from header if available
        const headerSessionId = response.headers.get('X-Session-Id');
        if (headerSessionId) {
            state.sessionId = headerSessionId;
        }

        // Stream response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.text) {
                            fullText += data.text;
                            updateAssistantMessage(assistantMsg, fullText);
                        }
                        if (data.done) {
                            if (data.session_id) {
                                state.sessionId = data.session_id;
                            }
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete chunks
                    }
                }
            }
        }

    } catch (error) {
        console.error('Chat error:', error);
        updateAssistantMessage(assistantMsg, `⚠️ Error: ${error.message}. Make sure the Groq API key is configured and files are uploaded.`);
    } finally {
        state.isStreaming = false;
        els.sendBtn.disabled = els.chatInput.value.trim().length === 0;
        els.sessionStatus.textContent = 'Ready';
    }
}

function appendMessage(role, content, isTyping = false) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = role === 'assistant' ? '⬡' : '👤';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    msg.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-body">
            <div class="message-bubble">
                ${isTyping ? `
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                ` : renderMarkdown(content)}
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;

    els.messages.appendChild(msg);
    scrollToBottom();

    return msg;
}

function updateAssistantMessage(msgEl, text) {
    const bubble = msgEl.querySelector('.message-bubble');
    bubble.innerHTML = renderMarkdown(text);
    scrollToBottom();
}

function scrollToBottom() {
    els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════════════
//  MARKDOWN RENDERER (lightweight)
// ═══════════════════════════════════════════════════════════════════════
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Tables
    html = renderTables(html);

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered lists
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');

    // Line breaks
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraphs
    if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
    }

    return html;
}

function renderTables(html) {
    const lines = html.split('\n');
    let inTable = false;
    let tableHtml = '';
    let result = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith('|') && line.endsWith('|')) {
            if (!inTable) {
                inTable = true;
                tableHtml = '<table>';
            }

            // Check if separator row
            if (/^\|[\s\-:|]+\|$/.test(line)) {
                continue;
            }

            const cells = line.split('|').filter(c => c.trim() !== '');
            const tag = tableHtml === '<table>' ? 'th' : 'td';
            tableHtml += '<tr>';
            cells.forEach(cell => {
                tableHtml += `<${tag}>${cell.trim()}</${tag}>`;
            });
            tableHtml += '</tr>';
        } else {
            if (inTable) {
                tableHtml += '</table>';
                result.push(tableHtml);
                tableHtml = '';
                inTable = false;
            }
            result.push(line);
        }
    }

    if (inTable) {
        tableHtml += '</table>';
        result.push(tableHtml);
    }

    return result.join('\n');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════════
//  SUMMARY MODAL (POST-based — fixes filename-with-dots issue)
// ═══════════════════════════════════════════════════════════════════════
async function openSummary(filename) {
    els.summaryModal.style.display = 'flex';
    els.summaryTitle.textContent = `📄 ${filename}`;
    els.summaryContent.innerHTML = `
        <div class="summary-loading">
            <div class="pulse-loader"></div>
            <p>Generating intelligent summary...</p>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/api/summary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                session_id: state.sessionId,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        els.summaryContent.innerHTML = renderMarkdown(data.summary);

    } catch (error) {
        console.error('Summary error:', error);
        els.summaryContent.innerHTML = `
            <div style="text-align:center; padding:24px; color:var(--error);">
                <p>⚠️ ${error.message}</p>
                <p style="font-size:0.8rem; color:var(--text-tertiary); margin-top:8px;">
                    Check that your Groq API key is valid and the server is running.
                </p>
            </div>
        `;
    }
}

function closeSummaryModal() {
    els.summaryModal.style.display = 'none';
}

// ═══════════════════════════════════════════════════════════════════════
//  SESSION MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════
async function clearSession() {
    if (state.sessionId) {
        try {
            await fetch(`${API_BASE}/api/session/${state.sessionId}`, {
                method: 'DELETE',
            });
        } catch (e) {
            console.warn('Failed to clear session on server:', e);
        }
    }

    // Reset state
    state.sessionId = '';
    state.files = [];
    state.totalChunks = 0;

    // Reset UI
    updateFileList();
    updateFilterSelect();
    updateChunkBadge();

    // Show welcome, hide messages
    els.welcomeScreen.style.display = 'flex';
    els.messages.style.display = 'none';
    els.messages.innerHTML = '';
    els.exportChatBtn.style.display = 'none';

    els.sessionStatus.textContent = 'Ready';
    showNotification('Session cleared', 'info');
}

// ═══════════════════════════════════════════════════════════════════════
//  EXPORT CHAT
// ═══════════════════════════════════════════════════════════════════════
function exportChat() {
    const messages = $$('.message');
    if (messages.length === 0) {
        showNotification('No messages to export', 'error');
        return;
    }

    let exportText = `Cortex AI - Chat Export\nDate: ${new Date().toLocaleString()}\nFiles: ${state.files.map(f => f.filename).join(', ') || 'None'}\n${'─'.repeat(60)}\n\n`;

    messages.forEach(msg => {
        const role = msg.classList.contains('user') ? 'You' : 'Cortex AI';
        const bubble = msg.querySelector('.message-bubble');
        const time = msg.querySelector('.message-time')?.textContent || '';
        const text = bubble?.innerText || '';
        exportText += `[${time}] ${role}:\n${text}\n\n`;
    });

    const blob = new Blob([exportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cortex-ai-chat-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification('Chat exported successfully', 'info');
}

// ═══════════════════════════════════════════════════════════════════════
//  NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════════
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-family: var(--font-sans);
        z-index: 200;
        animation: messageIn 0.3s ease-out;
        backdrop-filter: blur(12px);
        border: 1px solid ${type === 'error' ? 'rgba(239,68,68,0.3)' : 'rgba(99,102,241,0.3)'};
        background: ${type === 'error' ? 'rgba(239,68,68,0.15)' : 'rgba(99,102,241,0.15)'};
        color: ${type === 'error' ? '#f87171' : '#a78bfa'};
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(10px)';
        notification.style.transition = '0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ═══════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════
function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}
