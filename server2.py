import os
import re
import time
import json
import difflib
import joblib
import requests
import feedparser
import logging
import subprocess
import threading
import asyncio
import websockets
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import assemblyai as aai
from requests.exceptions import RequestException
import shutil

# Placeholder import for EnsembleModel (assumed to exist in models.py)
from models import EnsembleModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask + Socket.IO initialization
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths to saved models
MODEL_PATH = os.path.join('saved_models', 'ensemble_model.pkl')
PIPELINE_PATH = os.path.join('saved_models', 'pipeline.pkl')

# Verify model and pipeline existence
if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError("Model or pipeline not found. Please train and save them first.")

# Load model and pipeline
ensemble_model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# AssemblyAI configuration
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
transcriber = aai.Transcriber(config=aai.TranscriptionConfig(language_code="en"))

# Paths to executables
YT_DLP_PATH = shutil.which("yt-dlp") or os.path.join(os.getcwd(), "yt-dlp.exe")
FFMPEG_PATH = shutil.which("ffmpeg") or os.path.join(os.getcwd(), "ffmpeg.exe")

# Verify executables exist
if not os.path.exists(YT_DLP_PATH):
    raise FileNotFoundError(f"yt-dlp not found at {YT_DLP_PATH}.")
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}.")

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

# Helper function: analyze text
def analyze_text(text):
    try:
        X = pipeline.transform([text])
        prediction = ensemble_model.predict(X)
        confidence = ensemble_model.predict_proba(X)[:, 1][0]
        return {
            'prediction': 'True' if prediction[0] == 1 else 'False',
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f'Error in analyze_text: {e}')
        return {'prediction': 'Error', 'confidence': 0.0}

# Helper function: scrape article
def scrape_article(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.ads', '#comments']):
            element.decompose()
        article = soup.find('article') or soup.find('main') or soup.find('body')
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p') if len(p.get_text(strip=True)) > 30]
        return ' '.join(paragraphs)
    except Exception as e:
        logger.error(f'Error scraping article: {e}')
        return None

# Cache manager
class CacheManager:
    def __init__(self):
        self.caches = {}
        self.initialize_caches()

    def initialize_caches(self):
        for cache_type in ['audio', 'news', 'analysis', 'liveStreams']:
            self.caches[cache_type] = {}

    def get(self, type, key):
        cache = self.caches.get(type, {})
        item = cache.get(key)
        if not item or (datetime.now() - item['timestamp']).total_seconds() > item['ttl']:
            return None
        return item['data']

    def set(self, type, key, data, ttl=300):
        if type not in self.caches:
            self.caches[type] = {}
        self.caches[type][key] = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl': ttl
        }

    def clear(self):
        self.caches = {}
        self.initialize_caches()

cache_manager = CacheManager()

# News sources configuration
NEWS_SOURCES = {
    'RSS_FEEDS': [
        {'url': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms', 'name': 'Times of India', 'reliability': 0.8},
        {'url': 'https://www.thehindu.com/news/national/feeder/default.rss', 'name': 'The Hindu', 'reliability': 0.85}
    ],
    'GNEWS': {
        'endpoint': 'https://gnews.io/api/v4/top-headlines',
        'params': {
            'country': 'in',
            'lang': 'en',
            'max': 10,
            'token': os.getenv('GNEWS_API_KEY')
        }
    }
}

def fetch_gnews_articles():
    api_key = os.getenv('GNEWS_API_KEY')
    if not api_key:
        logger.warning('GNEWS_API_KEY is not set')
        return []
    try:
        response = requests.get(NEWS_SOURCES['GNEWS']['endpoint'], params=NEWS_SOURCES['GNEWS']['params'], timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return articles
    except RequestException as e:
        if e.response and e.response.status_code == 403:
            logger.error('GNews API error: Invalid API key')
        else:
            logger.error(f'GNews API error: {e}')
        return []

def fetch_trending_news():
    cached_news = cache_manager.get('news', 'trending')
    if cached_news:
        return cached_news
    try:
        rss_results = []
        for source in NEWS_SOURCES['RSS_FEEDS']:
            feed = feedparser.parse(source['url'])
            for item in feed.entries[:10]:
                rss_results.append({
                    'title': item.title,
                    'description': item.get('description', item.get('summary', '')),
                    'url': item.link,
                    'source': source['name'],
                    'reliability': source['reliability'],
                    'published': item.get('published', '')
                })
        gnews_results = fetch_gnews_articles()
        all_news = rss_results + gnews_results
        unique_news = []
        for current in all_news:
            is_duplicate = any(difflib.SequenceMatcher(None, item['title'], current['title']).ratio() > 0.8 for item in unique_news)
            if not is_duplicate:
                text = f"{current['title']} {current.get('description', '')}"
                analysis = analyze_text(text)
                unique_news.append({**current, 'analysis': analysis})
        result = unique_news[:15]
        cache_manager.set('news', 'trending', result)
        return result
    except Exception as e:
        logger.error(f'Error fetching trending news: {e}')
        return []

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-article', methods=['POST'])
def analyze_article():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "Article URL is required"}), 400
    article_text = scrape_article(url)
    if not article_text:
        return jsonify({"error": "Failed to extract article content"}), 500
    analysis = analyze_text(article_text)
    return jsonify({'text': article_text, 'analysis': analysis, 'success': True})

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text_route():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    analysis = analyze_text(text)
    return jsonify({'text': text, 'analysis': analysis, 'success': True})

@app.route('/api/trending-news', methods=['GET'])
def trending_news_route():
    return jsonify(fetch_trending_news())

@app.route('/api/news-stream')
def news_stream_route():
    def generate():
        while True:
            news = fetch_trending_news()
            yield f"data: {json.dumps(news)}\n\n"
            time.sleep(10)
    return app.response_class(generate(), mimetype='text/event-stream')

# Socket.IO for live streaming and transcription
active_streams = {}

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]
    logger.info('Client disconnected')

@socketio.on('start_live')
def handle_start_live(data):
    video_url = data.get('url')
    sid = request.sid
    if not video_url:
        emit('error', {'error': 'Video URL is required'})
        return

    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()

    stop_event = threading.Event()
    thread = threading.Thread(target=start_live_stream, args=(sid, video_url, stop_event))
    thread.start()
    active_streams[sid] = {'thread': thread, 'stop_event': stop_event}

def start_live_stream(sid, video_url, stop_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(handle_websocket(sid, video_url, stop_event))

async def handle_websocket(sid, video_url, stop_event):
    process = None
    ffmpeg_process = None
    try:
        headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
        logger.info("Connecting to WebSocket...")
        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?intent=transcription',
            extra_headers=headers
        ) as websocket:
            logger.info("WebSocket connected")
            setup_payload = {
                "type": "transcription_session.update",
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "prompt": "",
                    "language": "en"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "input_audio_noise_reduction": {
                    "type": "near_field"
                },
                "include": [
                    "item.input_audio_transcription.logprobs"
                ]
            }
            await websocket.send(json.dumps(setup_payload))

            process = subprocess.Popen(
                [YT_DLP_PATH, '-x', '--audio-format', 'wav', '--output', '-', '--no-playlist', '--live-from-start', video_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ffmpeg_process = subprocess.Popen(
                [FFMPEG_PATH, '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
                stdin=process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            async def send_audio():
                while not stop_event.is_set():
                    chunk = ffmpeg_process.stdout.read(4096)
                    if not chunk:
                        error = ffmpeg_process.stderr.read().decode()
                        if error:
                            logger.error(f"FFmpeg error: {error}")
                            socketio.emit('error', {'error': f"Stream failed: {error}"}, room=sid)
                        break
                    base64_chunk = base64.b64encode(chunk).decode('utf-8')
                    audio_payload = {
                        "type": "input_audio_buffer.append",
                        "audio": base64_chunk
                    }
                    await websocket.send(json.dumps(audio_payload))

            async def receive_messages():
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get('type') == 'transcription':
                            text = data.get('text', '')
                            analysis = analyze_text(text)
                            socketio.emit('transcription', {
                                'text': text,
                                'analysis': analysis,
                                'timestamp': datetime.now().isoformat()
                            }, room=sid)
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_messages())
            await asyncio.wait([send_task, receive_task], return_when=asyncio.FIRST_COMPLETED)

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"WebSocket connection failed: {e}")
        socketio.emit('error', {'error': f"WebSocket connection failed: {e}"}, room=sid)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        socketio.emit('error', {'error': str(e)}, room=sid)
    finally:
        if process:
            process.terminate()
        if ffmpeg_process:
            ffmpeg_process.terminate()

# Transcription helper functions
def get_video_duration(url):
    try:
        result = subprocess.run(
            [YT_DLP_PATH, '--dump-json', url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            raise Exception(f"yt-dlp failed: {result.stderr}")
        metadata = json.loads(result.stdout)
        duration = metadata.get('duration')
        if duration is None:
            raise Exception("Duration not found in metadata")
        return duration
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        return None

def transcribe_and_analyze(sid, video_url):
    try:
        # Extract audio
        process = subprocess.Popen(
            [YT_DLP_PATH, '-x', '--audio-format', 'mp3', '--format', 'bestaudio', '--output', '-', video_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        audio_data, error = process.communicate(timeout=120)
        if process.returncode != 0:
            raise Exception(f'yt-dlp failed: {error.decode()}')

        # Upload to AssemblyAI
        audio_file = transcriber.upload_file(audio_data)
        transcript = transcriber.transcribe(audio_file)
        if not transcript.text:
            raise Exception("Empty transcription")

        # Analyze text
        analysis = analyze_text(transcript.text)

        # Cache result
        result = {'text': transcript.text, 'analysis': analysis}
        cache_manager.set('audio', video_url, result)

        # Emit result
        socketio.emit('transcription_complete', result, room=sid)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        socketio.emit('transcription_error', {'error': str(e)}, room=sid)

@socketio.on('transcribe_recorded')
def handle_transcribe_recorded(data):
    video_url = data.get('video_url')
    if not video_url:
        emit('transcription_error', {'error': 'Video URL is required'})
        return

    sid = request.sid

    # Check cache
    cached = cache_manager.get('audio', video_url)
    if cached:
        emit('transcription_complete', {'text': cached['text'], 'analysis': cached['analysis']})
        return

    # Get duration
    duration = get_video_duration(video_url)
    if duration is None:
        emit('transcription_error', {'error': 'Could not determine video duration'})
        return
    if duration > 600:  # 10 minutes
        emit('transcription_error', {'error': 'Video is too long (>10 minutes)'})
        return

    # Start background thread
    threading.Thread(target=transcribe_and_analyze, args=(sid, video_url)).start()
    emit('transcription_started', {'message': 'Transcription started'})

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)

#updated python server2.py
# import os
# import re
# import time
# import json
# import difflib
# import joblib
# import requests
# import feedparser
# import logging
# import subprocess
# import io
# import base64
# import threading
# import asyncio
# import websockets
# from bs4 import BeautifulSoup
# from datetime import datetime
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify, Response, render_template
# from flask_socketio import SocketIO, emit
# import assemblyai as aai
# from requests.exceptions import RequestException
# import signal
# import shutil

# # Placeholder import for EnsembleModel (assumed to exist in models.py)
# from models import EnsembleModel

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Flask + Socket.IO initialization
# app = Flask(__name__, static_folder='static', template_folder='templates')
# app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Paths to saved models
# MODEL_PATH = os.path.join('saved_models', 'ensemble_model.pkl')
# PIPELINE_PATH = os.path.join('saved_models', 'pipeline.pkl')

# # Verify model and pipeline existence
# if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
#     raise FileNotFoundError("Model or pipeline not found. Please train and save them first.")

# # Load model and pipeline
# ensemble_model = joblib.load(MODEL_PATH)
# pipeline = joblib.load(PIPELINE_PATH)

# # AssemblyAI configuration
# aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
# transcriber = aai.Transcriber(config=aai.TranscriptionConfig(language_code="en"))

# # Paths to executables (adjust these if necessary)
# YT_DLP_PATH = shutil.which("yt-dlp") or os.path.join(os.getcwd(), "yt-dlp.exe")
# FFMPEG_PATH = shutil.which("ffmpeg") or os.path.join(os.getcwd(), "ffmpeg.exe")

# # Verify executables exist
# if not os.path.exists(YT_DLP_PATH):
#     raise FileNotFoundError(f"yt-dlp not found at {YT_DLP_PATH}. Please install it or update the path.")
# if not os.path.exists(FFMPEG_PATH):
#     raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}. Please install it or update the path.")

# # OpenAI API key
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY is not set in environment variables")

# # Helper function: analyze text
# def analyze_text(text):
#     try:
#         X = pipeline.transform([text])
#         prediction = ensemble_model.predict(X)
#         confidence = ensemble_model.predict_proba(X)[:, 1][0]
#         return {
#             'prediction': 'True' if prediction[0] == 1 else 'False',
#             'confidence': float(confidence)
#         }
#     except Exception as e:
#         logger.error(f'Error in analyze_text: {e}')
#         return {'prediction': 'Error', 'confidence': 0.0}

# # Helper function: scrape article
# def scrape_article(url):
#     try:
#         response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
#         for element in soup(['script', 'style', 'nav', 'header', 'footer', '.ads', '#comments']):
#             element.decompose()
#         article = soup.find('article') or soup.find('main') or soup.find('body')
#         paragraphs = [p.get_text(strip=True) for p in article.find_all('p') if len(p.get_text(strip=True)) > 30]
#         return ' '.join(paragraphs)
#     except Exception as e:
#         logger.error(f'Error scraping article: {e}')
#         return None

# # Cache manager
# class CacheManager:
#     def __init__(self):
#         self.caches = {}
#         self.initialize_caches()

#     def initialize_caches(self):
#         for cache_type in ['audio', 'news', 'analysis', 'liveStreams']:
#             self.caches[cache_type] = {}

#     def get(self, type, key):
#         cache = self.caches.get(type, {})
#         item = cache.get(key)
#         if not item or (datetime.now() - item['timestamp']).total_seconds() > item['ttl']:
#             return None
#         return item['data']

#     def set(self, type, key, data, ttl=300):
#         if type not in self.caches:
#             self.caches[type] = {}
#         self.caches[type][key] = {
#             'data': data,
#             'timestamp': datetime.now(),
#             'ttl': ttl
#         }

#     def clear(self):
#         self.caches = {}
#         self.initialize_caches()

# cache_manager = CacheManager()

# # News sources configuration
# NEWS_SOURCES = {
#     'RSS_FEEDS': [
#         {'url': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms', 'name': 'Times of India', 'reliability': 0.8},
#         {'url': 'https://www.thehindu.com/news/national/feeder/default.rss', 'name': 'The Hindu', 'reliability': 0.85}
#     ],
#     'GNEWS': {
#         'endpoint': 'https://gnews.io/api/v4/top-headlines',
#         'params': {
#             'country': 'in',
#             'lang': 'en',
#             'max': 10,
#             'token': os.getenv('GNEWS_API_KEY')
#         }
#     }
# }

# def fetch_gnews_articles():
#     api_key = os.getenv('GNEWS_API_KEY')
#     if not api_key or not re.match(r'^[a-zA-Z0-9]{32}$', api_key):
#         logger.error('Invalid or missing GNEWS_API_KEY')
#         return []
#     try:
#         response = requests.get(NEWS_SOURCES['GNEWS']['endpoint'], params=NEWS_SOURCES['GNEWS']['params'], timeout=10)
#         response.raise_for_status()
#         articles = response.json().get('articles', [])
#         return articles
#     except RequestException as e:
#         logger.error(f'GNews API error: {e}')
#         return []

# def fetch_trending_news():
#     cached_news = cache_manager.get('news', 'trending')
#     if cached_news:
#         return cached_news
#     try:
#         rss_results = []
#         for source in NEWS_SOURCES['RSS_FEEDS']:
#             feed = feedparser.parse(source['url'])
#             for item in feed.entries[:10]:
#                 rss_results.append({
#                     'title': item.title,
#                     'description': item.get('description', item.get('summary', '')),
#                     'url': item.link,
#                     'source': source['name'],
#                     'reliability': source['reliability'],
#                     'published': item.get('published', '')
#                 })
#         gnews_results = fetch_gnews_articles()
#         all_news = rss_results + gnews_results
#         unique_news = []
#         for current in all_news:
#             is_duplicate = any(difflib.SequenceMatcher(None, item['title'], current['title']).ratio() > 0.8 for item in unique_news)
#             if not is_duplicate:
#                 text = f"{current['title']} {current.get('description', '')}"
#                 analysis = analyze_text(text)
#                 unique_news.append({**current, 'analysis': analysis})
#         result = unique_news[:15]
#         cache_manager.set('news', 'trending', result)
#         return result
#     except Exception as e:
#         logger.error(f'Error fetching trending news: {e}')
#         return []

# # Flask routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/transcribe-recorded', methods=['POST'])
# def transcribe_recorded_route():
#     data = request.get_json()
#     video_url = data.get('video_url')
#     if not video_url:
#         return jsonify({"error": "Video URL is required"}), 400

#     max_retries = 3
#     retry_delay = 5  # seconds

#     for attempt in range(max_retries):
#         try:
#             process = subprocess.Popen(
#                 [YT_DLP_PATH, '-x', '--audio-format', 'mp3', '--output', '-', '--no-playlist', video_url],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#             audio_data, error = process.communicate(timeout=120)  # 2-minute timeout
#             if process.returncode != 0:
#                 raise Exception(f'yt-dlp failed: {error.decode()}')

#             audio_file = transcriber.upload_file(audio_data)
#             transcript = transcriber.transcribe(audio_file)
#             if not transcript.text:
#                 raise Exception("Transcription returned empty text")
            
#             analysis = analyze_text(transcript.text)
#             return jsonify({'text': transcript.text, 'analysis': analysis, 'success': True})

#         except subprocess.TimeoutExpired:
#             process.terminate()
#             logger.error(f"Transcription timed out on attempt {attempt + 1}/{max_retries}")
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay)
#                 continue
#             return jsonify({"error": "Transcription timed out after retries"}), 500

#         except Exception as e:
#             logger.error(f'Transcription error on attempt {attempt + 1}/{max_retries}: {e}')
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay)
#                 continue
#             return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

# @app.route('/api/analyze-article', methods=['POST'])
# def analyze_article():
#     data = request.get_json()
#     url = data.get('url')
#     if not url:
#         return jsonify({"error": "Article URL is required"}), 400
#     article_text = scrape_article(url)
#     if not article_text:
#         return jsonify({"error": "Failed to extract article content"}), 500
#     analysis = analyze_text(article_text)
#     return jsonify({'text': article_text, 'analysis': analysis, 'success': True})

# @app.route('/api/analyze-text', methods=['POST'])
# def analyze_text_route():
#     data = request.get_json()
#     text = data.get('text', '')
#     if not text:
#         return jsonify({"error": "Text is required"}), 400
#     analysis = analyze_text(text)
#     return jsonify({'text': text, 'analysis': analysis, 'success': True})

# @app.route('/api/trending-news', methods=['GET'])
# def trending_news_route():
#     return jsonify(fetch_trending_news())

# @app.route('/api/news-stream')
# def news_stream_route():
#     def generate():
#         while True:
#             news = fetch_trending_news()
#             yield f"data: {json.dumps(news)}\n\n"
#             time.sleep(10)
#     return Response(generate(), mimetype='text/event-stream')

# # Socket.IO for live streaming
# active_streams = {}

# @socketio.on('connect')
# def handle_connect():
#     logger.info('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     sid = request.sid
#     if sid in active_streams:
#         active_streams[sid]['stop_event'].set()
#         active_streams[sid]['thread'].join()
#         del active_streams[sid]
#     logger.info('Client disconnected')

# @socketio.on('start_live')
# def handle_start_live(data):
#     video_url = data.get('url')
#     sid = request.sid
#     if not video_url:
#         emit('error', {'error': 'Video URL is required'})
#         return

#     if sid in active_streams:
#         active_streams[sid]['stop_event'].set()
#         active_streams[sid]['thread'].join()

#     stop_event = threading.Event()
#     thread = threading.Thread(target=start_live_stream, args=(sid, video_url, stop_event))
#     thread.start()
#     active_streams[sid] = {'thread': thread, 'stop_event': stop_event}

# def start_live_stream(sid, video_url, stop_event):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(handle_websocket(sid, video_url, stop_event))
#     loop.close()

# async def handle_websocket(sid, video_url, stop_event):
#     process = None
#     ffmpeg_process = None
#     try:
#         headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
#         async with websockets.connect(
#             'wss://api.openai.com/v1/realtime?intent=transcription',
#             extra_headers=headers
#         ) as websocket:
#             setup_payload = {
#                 "type": "transcription_session.update",
#                 "input_audio_format": "pcm16",
#                 "input_audio_transcription": {
#                     "model": "whisper-1",
#                     "prompt": "",
#                     "language": "en"
#                 },
#                 "turn_detection": {
#                     "type": "server_vad",
#                     "threshold": 0.5,
#                     "prefix_padding_ms": 300,
#                     "silence_duration_ms": 500
#                 },
#                 "input_audio_noise_reduction": {
#                     "type": "near_field"
#                 },
#                 "include": [
#                     "item.input_audio_transcription.logprobs"
#                 ]
#             }
#             await websocket.send(json.dumps(setup_payload))

#             # Start yt-dlp asynchronously
#             process = await asyncio.create_subprocess_exec(
#                 YT_DLP_PATH, '-x', '--audio-format', 'wav', '--output', '-', '--no-playlist', '--live-from-start', video_url,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )

#             # Start FFmpeg asynchronously, reading from process.stdout
#             ffmpeg_process = await asyncio.create_subprocess_exec(
#                 FFMPEG_PATH, '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1',
#                 stdin=process.stdout,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )

#             async def send_audio():
#                 while not stop_event.is_set():
#                     chunk = await ffmpeg_process.stdout.read(4096)
#                     if not chunk:
#                         err_bytes = await ffmpeg_process.stderr.read()
#                         err_msg = err_bytes.decode() if err_bytes else ''
#                         if err_msg:
#                             logger.error(f"FFmpeg error: {err_msg}")
#                             socketio.emit('error', {'error': f"Stream failed: {err_msg}"}, room=sid)
#                         break
#                     base64_chunk = base64.b64encode(chunk).decode('utf-8')
#                     audio_payload = {
#                         "type": "input_audio_buffer.append",
#                         "audio": base64_chunk
#                     }
#                     await websocket.send(json.dumps(audio_payload))
#                 await websocket.send(json.dumps({"type": "input_audio_buffer.end"}))

#             async def receive_messages():
#                 while True:
#                     try:
#                         message = await websocket.recv()
#                         data = json.loads(message)
#                         if data.get('type') == 'transcription':
#                             text = data.get('text', '')
#                             analysis = analyze_text(text)
#                             socketio.emit('transcription', {
#                                 'text': text,
#                                 'analysis': analysis,
#                                 'timestamp': datetime.now().isoformat()
#                             }, room=sid)
#                     except websockets.exceptions.ConnectionClosed:
#                         logger.info("WebSocket connection closed")
#                         break

#             send_task = asyncio.create_task(send_audio())
#             receive_task = asyncio.create_task(receive_messages())
#             done, pending = await asyncio.wait(
#                 [send_task, receive_task],
#                 return_when=asyncio.FIRST_COMPLETED
#             )
#             for task in pending:
#                 task.cancel()

#     except Exception as e:
#         logger.error(f"WebSocket error: {e}")
#         socketio.emit('error', {'error': str(e)}, room=sid)
#     finally:
#         if process:
#             process.terminate()
#         if ffmpeg_process:
#             ffmpeg_process.terminate()

# if __name__ == '__main__':
#     PORT = int(os.getenv('PORT', 3000))
#     socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
