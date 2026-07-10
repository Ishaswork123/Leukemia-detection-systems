document.addEventListener('DOMContentLoaded', () => {
    // --- Element refs ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImg = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');

    const resultsEmpty = document.getElementById('results-empty');
    const resultsContent = document.getElementById('results-content');
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');
    const heatmapBox = document.getElementById('heatmap-box');
    const heatmapImage = document.getElementById('heatmap-image');
    const showHeatmapBtn = document.getElementById('show-heatmap-btn');

    const errorAlert = document.getElementById('error-alert');
    const errorAlertText = document.getElementById('error-alert-text');
    const errorAlertClose = document.getElementById('error-alert-close');

    let currentFile = null;
    let errorTimeout = null;

    // --- Error alert helpers ---
    function showError(message) {
        errorAlertText.textContent = message;
        errorAlert.classList.remove('hidden');
        clearTimeout(errorTimeout);
        errorTimeout = setTimeout(() => errorAlert.classList.add('hidden'), 6000);
    }
    errorAlertClose.addEventListener('click', () => errorAlert.classList.add('hidden'));

    // --- Results helpers ---
    function clearResults() {
        resultsContent.classList.add('hidden');
        resultsEmpty.classList.remove('hidden');
        predictionText.textContent = '--';
        confidenceText.textContent = '--';
        heatmapBox.style.display = 'none';
        heatmapImage.classList.add('hidden');
        heatmapImage.src = '';
        showHeatmapBtn.classList.remove('hidden');
    }

    function renderResults(data) {
        resultsEmpty.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        let confStr = data.confidence;
        if (!isNaN(parseFloat(data.confidence))) {
            confStr = (parseFloat(data.confidence) * 100).toFixed(1) + '%';
        }

        predictionText.textContent = data.prediction || 'Unknown';
        confidenceText.textContent = confStr;

        if (data.heatmap_url) {
            heatmapBox.style.display = 'block';
            heatmapImage.classList.add('hidden'); // hidden until user clicks "Show"
            showHeatmapBtn.classList.remove('hidden');
            showHeatmapBtn.dataset.src = data.heatmap_url + '?t=' + Date.now();
        } else {
            heatmapBox.style.display = 'none';
        }
    }

    showHeatmapBtn.addEventListener('click', () => {
        const src = showHeatmapBtn.dataset.src;
        if (!src) return;
        heatmapImage.src = src;
        heatmapImage.classList.remove('hidden');
        showHeatmapBtn.classList.add('hidden');
    });

    // --- Upload / drop zone ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please select a valid image file.');
            return;
        }
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewContainer.style.display = 'block';
            dropZone.style.display = 'none';
            analyzeBtn.disabled = false;

            // Wipe any previous/fake results the moment a new file is chosen
            clearResults();
        };
        reader.readAsDataURL(file);
    }

    // --- Analyze ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        clearResults(); // never show stale/fake results while a request is in flight

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // FastAPI's HTTPException body is {"detail": "..."}
                let detail = 'Something went wrong. Please try again.';
                try {
                    const errBody = await response.json();
                    if (errBody && errBody.detail) detail = errBody.detail;
                } catch (_) { /* keep default message */ }

                clearResults();
                showError(detail);
                return;
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            clearResults();
            showError('Network error: could not reach the server. Please try again.');
            console.error('Analysis error:', error);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Scan';
        }
    });

    // --- Chatbot toggle ---
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotPanel = document.getElementById('chatbot-panel');
    const chatbotClose = document.getElementById('chatbot-close');

    chatbotToggle.addEventListener('click', () => {
        chatbotPanel.classList.toggle('hidden');
    });
    chatbotClose.addEventListener('click', () => chatbotPanel.classList.add('hidden'));

    // --- Chatbot messaging ---
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-msg-btn');

    function appendMessage(sender, text, isSystem = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isSystem ? 'system-msg' : (sender === 'user' ? 'user-msg' : 'assistant-msg')}`;

        const contentP = document.createElement('p');
        contentP.textContent = text;
        msgDiv.appendChild(contentP);

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        chatInput.value = '';
        chatInput.disabled = true;
        sendBtn.disabled = true;

        const loadingId = Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant-msg';
        loadingDiv.id = `loading-${loadingId}`;
        loadingDiv.innerHTML = '<p>...</p>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await response.json();

            document.getElementById(`loading-${loadingId}`)?.remove();

            if (response.ok) {
                appendMessage('assistant', data.response);
            } else {
                appendMessage('system', data.detail || 'Failed to get response', true);
            }
        } catch (error) {
            document.getElementById(`loading-${loadingId}`)?.remove();
            appendMessage('system', 'Network error. Could not reach clinical assistant.', true);
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Initial state
    clearResults();
});