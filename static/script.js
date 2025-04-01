// Configuration
const CONFIG = {
    API_URL: window.location.origin,
    ERROR_DISPLAY_DURATION: 5000
};

// DOM Elements
const elements = {
    form: document.getElementById('transcriptionForm'),
    videoType: document.getElementById('videoType'),
    videoUrl: document.getElementById('videoUrl'),
    language: document.getElementById('language'),
    startButton: document.getElementById('startTranscription'),
    stopButton: document.getElementById('stopTranscription'),
    loadingMessage: document.getElementById('loadingMessage'),
    transcriptionResult: document.getElementById('transcriptionResult'),
    factCheckResult: document.getElementById('factCheckResult'),
    trendingNews: document.getElementById('trendingNews'),
    textInput: document.getElementById('textInput'),
    articleUrl: document.getElementById('articleUrl'),
    confidenceChart: document.getElementById('confidenceChart'),
    refreshNews: document.getElementById('refreshNews')
};

// Socket.IO connection
const socket = io(window.location.origin);

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Socket.IO connected');
    UI.showStatus('Connected to transcription service');
});
socket.on('transcription', (data) => {
    UI.updateTranscription(data.text, true);
    if (data.analysis) UI.updateFactCheck(data.analysis);
});
socket.on('error', (data) => UI.showError(data.error));

// UI functions
const UI = {
    showError(message, duration = CONFIG.ERROR_DISPLAY_DURATION) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg text-sm z-50';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), duration);
    },
    showStatus(message) {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'fixed top-4 right-4 bg-green-100 text-green-700 px-4 py-2 rounded-lg shadow-lg text-sm z-50';
        statusDiv.textContent = message;
        document.body.appendChild(statusDiv);
        setTimeout(() => statusDiv.remove(), 3000);
    },
    showLoading(show) {
        elements.loadingMessage.classList.toggle('hidden', !show);
        elements.startButton.disabled = show;
    },
    updateTranscription(text, append = false) {
        if (append) {
            const p = document.createElement('p');
            p.textContent = text;
            p.className = 'mb-2 p-2 bg-white rounded text-sm';
            elements.transcriptionResult.appendChild(p);
            elements.transcriptionResult.scrollTop = elements.transcriptionResult.scrollHeight;
        } else {
            elements.transcriptionResult.innerHTML = `<p class="mb-2 p-2 bg-white rounded text-sm">${text}</p>`;
        }
    },
    clearTranscription() {
        elements.transcriptionResult.innerHTML = '';
        elements.factCheckResult.innerHTML = '';
        Plotly.purge(elements.confidenceChart);
    },
    updateFactCheck(analysis) {
        if (!analysis) return;
        const { confidence, prediction } = analysis;
        elements.factCheckResult.innerHTML = `
            <div class="p-4 rounded-lg ${confidence > 0.7 ? 'text-green-600 bg-green-50' : confidence > 0.5 ? 'text-yellow-600 bg-yellow-50' : 'text-red-600 bg-red-50'}">
                <div class="flex justify-between">
                    <span>Credibility Score:</span>
                    <span>${(confidence * 100).toFixed(2)}%</span>
                </div>
                <div>Prediction: ${prediction}</div>
            </div>
        `;
        UI.createConfidenceChart(confidence);
    },
    createConfidenceChart(confidence) {
        const data = [{
            type: 'indicator',
            mode: 'gauge+number',
            value: confidence * 100,
            gauge: {
                axis: { range: [0, 100] },
                bar: { color: `hsl(${confidence * 120}, 70%, 50%)` },
                steps: [
                    { range: [0, 50], color: '#fee2e2' },
                    { range: [50, 70], color: '#fef3c7' },
                    { range: [70, 100], color: '#dcfce7' }
                ]
            }
        }];
        const layout = { width: elements.confidenceChart.offsetWidth, height: 250, margin: { t: 25, r: 25, l: 25, b: 25 } };
        Plotly.newPlot(elements.confidenceChart, data, layout, { responsive: true, displayModeBar: false });
    },
    updateTrendingNews(news) {
        if (!Array.isArray(news)) return;
        elements.trendingNews.innerHTML = news.map(article => `
            <div class="mb-3 p-3 bg-white rounded-lg shadow-sm">
                <h3 class="text-base font-semibold">${article.title}</h3>
                <p class="text-sm text-gray-600 mt-2">${article.description || ''}</p>
                <div class="mt-3 flex justify-between">
                    <span class="${article.analysis.confidence > 0.7 ? 'text-green-600' : article.analysis.confidence > 0.5 ? 'text-yellow-600' : 'text-red-600'}">
                        Credibility: ${(article.analysis.confidence * 100).toFixed(2)}%
                    </span>
                    <a href="${article.url}" target="_blank" class="text-blue-500 text-xs">Read more</a>
                </div>
            </div>
        `).join('');
    }
};

// API functions
const API = {
    async transcribeRecorded(videoUrl, language) {
        const response = await fetch(`${CONFIG.API_URL}/api/transcribe-recorded`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_url: videoUrl, language })
        });
        if (!response.ok) throw new Error('Failed to transcribe video');
        return response.json();
    },
    async analyzeArticle(url, language) {
        const response = await fetch(`${CONFIG.API_URL}/api/analyze-article`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, language })
        });
        if (!response.ok) throw new Error('Failed to analyze article');
        return response.json();
    },
    async analyzeText(text, language) {
        const response = await fetch(`${CONFIG.API_URL}/api/analyze-text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language })
        });
        if (!response.ok) throw new Error('Failed to analyze text');
        return response.json();
    },
    async fetchTrendingNews() {
        const response = await fetch(`${CONFIG.API_URL}/api/trending-news`);
        if (!response.ok) throw new Error('Failed to fetch news');
        return response.json();
    },
    setupNewsStream() {
        const eventSource = new EventSource(`${CONFIG.API_URL}/api/news-stream`);
        eventSource.onmessage = (event) => UI.updateTrendingNews(JSON.parse(event.data));
        eventSource.onerror = () => {
            eventSource.close();
            setTimeout(API.setupNewsStream, 2000);
        };
    }
};

// Transcription Manager
const TranscriptionManager = {
    async handleRecorded(videoUrl, language) {
        UI.showLoading(true);
        try {
            const data = await API.transcribeRecorded(videoUrl, language);
            UI.updateTranscription(data.text);
            UI.updateFactCheck(data.analysis);
            UI.showStatus('Transcription completed');
        } catch (error) {
            UI.showError(error.message);
        } finally {
            UI.showLoading(false);
        }
    },
    startLive(videoUrl, language) {
        socket.emit('start_live', { url: videoUrl, language });
        elements.stopButton.classList.remove('hidden');
        elements.startButton.classList.add('hidden');
        UI.showStatus('Live transcription started');
    },
    stopLive() {
        socket.disconnect();
        socket.connect(); // Reconnect for future use
        elements.stopButton.classList.add('hidden');
        elements.startButton.classList.remove('hidden');
        UI.showStatus('Live transcription stopped');
    },
    async handleArticleAnalysis(url, language) {
        UI.showLoading(true);
        try {
            const data = await API.analyzeArticle(url, language);
            UI.updateTranscription(data.text);
            UI.updateFactCheck(data.analysis);
            UI.showStatus('Article analysis completed');
        } catch (error) {
            UI.showError(error.message);
        } finally {
            UI.showLoading(false);
        }
    },
    async handleTextAnalysis(text, language) {
        UI.showLoading(true);
        try {
            const data = await API.analyzeText(text, language);
            UI.updateTranscription(data.text);
            UI.updateFactCheck(data.analysis);
            UI.showStatus('Text analysis completed');
        } catch (error) {
            UI.showError(error.message);
        } finally {
            UI.showLoading(false);
        }
    }
};

// Event Handlers
const EventHandlers = {
    handleVideoTypeChange(e) {
        const type = e.target.value;
        const urlContainer = document.getElementById('urlContainer');
        const textContainer = document.getElementById('textContainer');
        const articleContainer = document.getElementById('articleContainer');
        urlContainer.classList.add('hidden');
        textContainer.classList.add('hidden');
        articleContainer.classList.add('hidden');
        if (type === 'text') textContainer.classList.remove('hidden');
        else if (type === 'article') articleContainer.classList.remove('hidden');
        else urlContainer.classList.remove('hidden');
        document.getElementById('startButtonText').textContent =
            type === 'text' ? 'Analyze Text' :
            type === 'article' ? 'Analyze Article' :
            type === 'live' ? 'Start Live Transcription' : 'Start Analysis';
        if (!elements.stopButton.classList.contains('hidden')) {
            TranscriptionManager.stopLive();
        }
    },
    handleFormSubmit(e) {
        e.preventDefault();
        const videoType = elements.videoType.value;
        const language = elements.language.value;
        UI.clearTranscription();
        if (videoType === 'recorded') TranscriptionManager.handleRecorded(elements.videoUrl.value, language);
        else if (videoType === 'live') TranscriptionManager.startLive(elements.videoUrl.value, language);
        else if (videoType === 'text') TranscriptionManager.handleTextAnalysis(elements.textInput.value, language);
        else if (videoType === 'article') TranscriptionManager.handleArticleAnalysis(elements.articleUrl.value, language);
    },
    handleRefreshNews() {
        API.fetchTrendingNews().then(UI.updateTrendingNews).catch(() => UI.showError('Failed to refresh news'));
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    elements.form.addEventListener('submit', EventHandlers.handleFormSubmit);
    elements.stopButton.addEventListener('click', TranscriptionManager.stopLive);
    elements.videoType.addEventListener('change', EventHandlers.handleVideoTypeChange);
    elements.refreshNews.addEventListener('click', EventHandlers.handleRefreshNews);
    API.fetchTrendingNews().then(UI.updateTrendingNews);
    API.setupNewsStream();
});