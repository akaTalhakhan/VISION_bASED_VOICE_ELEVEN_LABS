## We optimize
import streamlit as st
import base64
import os
import time
import requests
import threading
import queue
import asyncio
import aiohttp
from threading import Lock
from tempfile import NamedTemporaryFile
from openai import OpenAI
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
import cv2
from pygame import mixer
import yt_dlp
import uuid
import io
from concurrent.futures import ThreadPoolExecutor
import atexit
from io import BytesIO
import pygame

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False

load_dotenv()


st.title("🎙️ TTS Narrator")

# 🔐 Sidebar for user-provided API keys
st.sidebar.header("🔐 API Key Configuration")

openai_input = st.sidebar.text_input("🧠 OpenAI API Key", type="password")
eleven_input = st.sidebar.text_input("🗣️ ElevenLabs API Key", type="password")

if openai_input:
    os.environ["OPENAI_API_KEY"] = openai_input
    client_openai = OpenAI(api_key=openai_input)

if eleven_input:
    os.environ["ELEVENLABS_API_KEY"] = eleven_input
    client_eleven = ElevenLabs(api_key=eleven_input)
    eleven_api_key = eleven_input


DEFAULT_VOICE_ID = "o1GYQOsPeFjZRhnXfhdg"

# Initialize session state safely
def init_session_state():
    """Initialize session state variables safely"""
    if 'cloned_voice_id' not in st.session_state:
        st.session_state.cloned_voice_id = None
    if 'cloned_voice_name' not in st.session_state:
        st.session_state.cloned_voice_name = None

# Call initialization
init_session_state()

CURRENT_VOICE_ID = None
CURRENT_VOICE_NAME = None

# Audio queue system with streaming support
audio_queue = queue.Queue()
audio_lock = Lock()
audio_thread_running = False
executor = ThreadPoolExecutor(max_workers=4)

# Description memory for preventing repetition
recent_descriptions = []
MAX_RECENT_DESCRIPTIONS = 3
last_response_time = 0
SILENCE_THRESHOLD = 8
current_prompt_level = 0

# Object tracking for better detection
last_detected_objects = []
last_hand_objects = []

# JPEG quality setting (reduced for better performance)
JPEG_QUALITY = 60



class ImprovedAudioStreamer:
    """Improved audio streamer with better file handling and validation"""
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.playing = False
        self.temp_files = []  # Track temp files for cleanup
        
    def stream_audio(self, audio_data):
        """Stream audio data with validation and improved file handling"""
        try:
            # Validate audio data
            if not audio_data or len(audio_data) < 1000:
                print(f"Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                return
                
            # Check if it's valid MP3 by looking for MP3 header
            if not audio_data.startswith(b'ID3') and not audio_data.startswith(b'\xff\xfb'):
                print("Audio data doesn't appear to be valid MP3")
                return
            
            # Create temp file with unique name
            temp_file = NamedTemporaryFile(delete=False, suffix=".mp3", prefix="audio_stream_")
            temp_file.write(audio_data)
            temp_file.flush()
            
            # Verify file was written correctly
            file_size = temp_file.tell()
            temp_file.close()
            
            if file_size < 1000:
                print(f"Audio file too small after writing: {file_size} bytes")
                os.unlink(temp_file.name)
                return
            
            # Add to cleanup tracking
            self.temp_files.append(temp_file.name)
            
            # Clean up old files if too many accumulate
            if len(self.temp_files) > 10:
                self.cleanup_old_files()
            
            audio_queue.put(temp_file.name)
            
        except Exception as e:
            print(f"Audio streaming error: {e}")
    
    def cleanup_old_files(self):
        """Clean up old temporary files"""
        files_to_remove = self.temp_files[:-5]  # Keep last 5
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                self.temp_files.remove(file_path)
            except:
                pass

    def play_audio_from_memory(self, audio_data):
        """Play audio directly from memory without temp files"""
        try:
            # Validate audio data first
            if not audio_data or len(audio_data) < 1000:
                print("Invalid audio data for memory playback")
                return
                
            sound = pygame.mixer.Sound(BytesIO(audio_data))
            sound.play()
            # Wait for sound to finish
            while pygame.mixer.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Memory audio playback error: {e}")
            # Fallback to file-based playback
            self.stream_audio(audio_data)

audio_streamer = ImprovedAudioStreamer()

def audio_manager():
    """Continuously plays audio from queue with streaming support"""
    global audio_thread_running
    pygame.mixer.init()
    
    while audio_thread_running:
        try:
            audio_path = audio_queue.get(timeout=1)
            
            # Check if file exists before trying to play
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                audio_queue.task_done()
                continue
            
            # Load and play audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Small delay to ensure mixer is done with the file
            time.sleep(0.2)
            
            # Now safely delete the file
            try:
                os.unlink(audio_path)
            except FileNotFoundError:
                pass  # File already deleted or moved
            except PermissionError:
                print(f"Permission error deleting {audio_path}")
            
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Audio playback error: {e}")
            audio_queue.task_done()

async def get_description_async(image_bytes, is_live=False):
    """Async function to get image description from OpenAI"""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    if is_live:
        prompt = get_adaptive_prompt()
    else:
        prompt = """Describe what you see in this image. Focus on the person's actions, expressions, and surroundings. If there are any objects visible, mention them naturally as part of the description."""
    
    loop = asyncio.get_event_loop()
    
    def call_openai():
        return client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=100
        )
    
    response = await loop.run_in_executor(executor, call_openai)
    return response.choices[0].message.content

async def generate_speech_async(text, voice_id, stream=False):
    """Async function to generate speech with proper error handling"""
    loop = asyncio.get_event_loop()
    
    def call_elevenlabs():
        try:
            # Use regular API call for more reliable audio
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json={
                    "text": text, 
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5
                    }
                },
                headers={
                    "xi-api-key": eleven_api_key,
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                audio_data = response.content
                # Validate that we have valid audio data
                if len(audio_data) > 1000:  # Basic check for reasonable file size
                    return audio_data
                else:
                    print(f"Audio data too small: {len(audio_data)} bytes")
                    return None
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("ElevenLabs API timeout")
            return None
        except requests.exceptions.RequestException as e:
            print(f"ElevenLabs API request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in ElevenLabs call: {e}")
            return None
    
    return await loop.run_in_executor(executor, call_elevenlabs)

async def describe_and_speak_async(image_bytes, voice_id, is_live=False):
    """Async main function combining description and speech generation"""
    global last_response_time
    
    try:
        # Get description asynchronously
        description = await get_description_async(image_bytes, is_live)
        
        # For live video, check similarity and timing
        if is_live:
            is_first_response = len(recent_descriptions) == 0
            
            if should_force_response() or is_first_response:
                pass
            elif is_description_similar(description):
                return
            
            add_to_recent_descriptions(description)
            last_response_time = time.time()
        
        # Generate speech asynchronously
        audio_data = await generate_speech_async(description, voice_id, stream=True)
        
        if is_live:
            # Stream audio for live video
            audio_streamer.stream_audio(audio_data)
        else:
            # For static images, we'll return the data to be handled by the main thread
            return description, audio_data
            
    except Exception as e:
        print(f"Error in describe_and_speak_async: {e}")
        return None, None

def describe_and_speak(image_bytes, is_live=False):
    """Wrapper function to run async code"""
    try:
        
        voice_id = CURRENT_VOICE_ID or (st.session_state.cloned_voice_id if 'cloned_voice_id' in st.session_state else None) or DEFAULT_VOICE_ID
        
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(describe_and_speak_async(image_bytes, voice_id, is_live))
            finally:
                loop.close()
        
        if is_live:
            # For live video, run in background thread
            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()
        else:
            # For static images, wait for result and display in main thread
            result = run_async()
            if result and result[0] and result[1]:
                description, audio_data = result
                st.write(description)
                st.audio(audio_data, format="audio/mp3")
            
    except Exception as e:
        print(f"Error in describe_and_speak: {e}")

def text_similarity(text1, text2):
    """Calculate similarity between two texts using simple word overlap"""
    if not text1 or not text2:
        return 0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0

def is_description_similar(new_description):
    """Check if new description is similar to recent ones"""
    global recent_descriptions
    
    if not recent_descriptions:
        return False
    
    for old_desc in recent_descriptions:
        similarity = text_similarity(new_description, old_desc)
        if similarity > 0.7:
            return True
    
    if len(new_description.split()) < 10:
        return True
        
    return False

def add_to_recent_descriptions(description):
    """Add description to recent memory"""
    global recent_descriptions
    
    recent_descriptions.append(description)
    if len(recent_descriptions) > MAX_RECENT_DESCRIPTIONS:
        recent_descriptions.pop(0)

def get_adaptive_prompt():
    """Get different types of prompts focused on natural object detection"""
    global current_prompt_level
    
    prompts = [
        "Describe what you see in this image. Focus on the person's actions and any visible objects.",
        "What is the person doing in this image? Describe their activity and surroundings.",
        "Describe the scene and environment. What can you see in the background and around the person?",
        "Describe what's happening in this moment. What does the person appear to be doing?",
        "Tell me what you observe in this image. Describe the person and their current situation."
    ]
    
    current_prompt_level = (current_prompt_level + 1) % len(prompts)
    return prompts[current_prompt_level]

def should_force_response():
    """Check if we should force a response due to silence"""
    global last_response_time
    return time.time() - last_response_time > SILENCE_THRESHOLD

def encode_image_optimized(image):
    """Encode image with optimized JPEG quality for faster processing"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes()

def clone_voice_from_audio(audio_data, voice_name_suffix=""):
    """Clone voice from audio data"""
    voice_name = f"Cloned_Voice_{voice_name_suffix}_{uuid.uuid4().hex[:8]}"
    
    response = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        files={'files': ('audio.mp3', audio_data, 'audio/mpeg')},
        data={
            'name': voice_name,
            'description': f"Voice cloned from uploaded audio",
            'labels': '{"accent": "cloned", "description": "uploaded", "age": "adult", "gender": "neutral", "use_case": "narration"}'
        },
        headers={'xi-api-key': eleven_api_key}
    )
    
    if response.status_code == 200:
        voice_data = response.json()
        return voice_data.get('voice_id'), voice_name
    else:
        st.error(f"Failed to clone voice: {response.text}")
        return None, None

def download_and_clone_voice(youtube_url):
    """Download and clone voice from YouTube URL"""
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': f'{temp_dir}/%(title)s.%(ext)s',
        'noplaylist': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        title = info.get('title', 'Unknown')
        ydl.download([youtube_url])
        
        for file in os.listdir(temp_dir):
            if file.endswith(('.mp3', '.m4a', '.webm')):
                audio_path = os.path.join(temp_dir, file)
                break
    
    voice_name = f"Cloned_{title[:30]}_{uuid.uuid4().hex[:8]}"
    
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    response = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        files={'files': ('audio.mp3', audio_data, 'audio/mpeg')},
        data={
            'name': voice_name,
            'description': f"Voice cloned from: {title}",
            'labels': '{"accent": "cloned", "description": "youtube", "age": "adult", "gender": "neutral", "use_case": "narration"}'
        },
        headers={'xi-api-key': eleven_api_key}
    )
    
    os.remove(audio_path)
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
        os.rmdir(temp_dir)
    
    if response.status_code == 200:
        voice_data = response.json()
        return voice_data.get('voice_id'), voice_name
    else:
        st.error(f"Failed to clone voice: {response.text}")
        return None, None

def cleanup_audio():
    """Clean up audio resources on exit"""
    global audio_thread_running
    audio_thread_running = False
    if hasattr(audio_streamer, 'cleanup_old_files'):
        audio_streamer.cleanup_old_files()

# Register cleanup on exit
atexit.register(cleanup_audio)

if WEBRTC_AVAILABLE:
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.last_time = 0
            self.processing_count = 0
            self.max_concurrent = 2  
            self.first_frame_processed = False
            self.start_time = None
            
            global audio_thread_running, last_response_time
            if not audio_thread_running:
                audio_thread_running = True
                last_response_time = time.time()
                threading.Thread(target=audio_manager, daemon=True).start()

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            
            if self.start_time is None:
                self.start_time = time.time()
            
            current_time = time.time()
            should_process = False
            
            if not self.first_frame_processed:
                should_process = True
                self.first_frame_processed = True
            elif (current_time - self.last_time >= 1.0) and self.processing_count < self.max_concurrent:
                should_process = True
            
            if should_process:
                self.last_time = current_time
                self.processing_count += 1
                
                def process():
                    try:
                        # Use optimized image encoding
                        image_bytes = encode_image_optimized(image.copy())
                        
                        # Get voice ID safely from global variables set in main thread
                        voice_id = CURRENT_VOICE_ID or DEFAULT_VOICE_ID
                        
                        # Run async processing
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(describe_and_speak_async(image_bytes, voice_id, is_live=True))
                        finally:
                            loop.close()
                            
                    except Exception as e:
                        print(f"Video processing error: {e}")
                    finally:
                        self.processing_count -= 1
                
                threading.Thread(target=process, daemon=True).start()
            
            return image

# UI Part:
st.header(" Voice Cloning")
voice_tab1, voice_tab2, voice_tab3 = st.tabs(["⬆️🎵 Upload Audio", "🎙️🔴🗣️Record Your Own Voice", "🎙️🧬📺YouTube"])

with voice_tab1:
    uploaded_audio = st.file_uploader("🔼🎵Upload audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        
        if st.button("Clone Voice from Upload", type="primary"):
            with st.spinner("Cloning voice..."):
                audio_data = uploaded_audio.read()
                voice_id, voice_name = clone_voice_from_audio(audio_data, "Upload")
                
                if voice_id:
                    st.session_state.cloned_voice_id = voice_id
                    st.session_state.cloned_voice_name = voice_name
                    st.success(f"Voice cloned: {voice_name}")
                    st.rerun()

with voice_tab2:
    if MIC_RECORDER_AVAILABLE:
        audio_data = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=False,
            use_container_width=True,
            key="mic_recorder"
        )
        
        if audio_data is not None:
            st.audio(audio_data['bytes'], format="audio/wav")
            
            if st.button("Clone Voice from Recording", type="primary"):
                with st.spinner("Cloning voice..."):
                    voice_id, voice_name = clone_voice_from_audio(audio_data['bytes'], "Recording")
                    
                    if voice_id:
                        st.session_state.cloned_voice_id = voice_id
                        st.session_state.cloned_voice_name = voice_name
                        st.success(f"Voice cloned: {voice_name}")
                        st.rerun()
    else:
        st.warning("Install `streamlit-mic-recorder` to enable recording")

with voice_tab3:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        youtube_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        if st.button("Clone Voice", type="primary"):
            if youtube_url:
                with st.spinner("Cloning from YouTube..."):
                    voice_id, voice_name = download_and_clone_voice(youtube_url)
                    
                    if voice_id:
                        st.session_state.cloned_voice_id = voice_id
                        st.session_state.cloned_voice_name = voice_name
                        st.success(f"Voice cloned: {voice_name}")
                        st.rerun()
            else:
                st.error("Please enter a YouTube URL")


if st.session_state.cloned_voice_id:
    st.success(f"Using cloned voice: **{st.session_state.cloned_voice_name}**")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Reset to Default Voice"):
            st.session_state.cloned_voice_id = None
            st.session_state.cloned_voice_name = None
            st.success("Reset to default voice")
            st.rerun()
    with col2:
        if st.button("Delete Cloned Voice"):
            try:
                delete_response = requests.delete(
                    f"https://api.elevenlabs.io/v1/voices/{st.session_state.cloned_voice_id}",
                    headers={'xi-api-key': eleven_api_key}
                )
                if delete_response.status_code == 200:
                    st.session_state.cloned_voice_id = None
                    st.session_state.cloned_voice_name = None
                    st.success("Voice deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete voice")
            except Exception as e:
                st.error(f"Error deleting voice: {str(e)}")
else:
    st.info("🎤🧑‍💻Using default voice")


st.info("Live video used aysnc , reduced image quality")

tab1, tab2, tab3 = st.tabs(["⬆️🖼️Upload Image", "📸Camera", "🔴📹Live Video"])

with tab1:
    uploaded_file = st.file_uploader("⬆️Upload", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Process and Speak", type="primary"):
            with st.spinner("Processing..."):
                describe_and_speak(uploaded_file.read())

with tab2:
    camera_image = st.camera_input("📸Take a picture")
    
    if camera_image is not None:
        if st.button("Process and Speak", key="cam", type="primary"):
            with st.spinner("Processing..."):
                describe_and_speak(camera_image.read())

with tab3:
    if WEBRTC_AVAILABLE:
        CURRENT_VOICE_ID = st.session_state.cloned_voice_id
        CURRENT_VOICE_NAME = st.session_state.cloned_voice_name
        
        st.info("**Fast Live Video with improveed response time **")
        
        webrtc_streamer(
            key="narrator",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    else:
        st.warning("Install `streamlit-webrtc` for live video")