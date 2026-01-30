# 🎙️ TTS Narrator

**Transform your visual world into spoken stories with AI-powered voice cloning**

A sophisticated multimodal AI application that combines **GPT-4o vision** with **ElevenLabs voice synthesis** to create personalized narrations of images and live video streams. Built with Streamlit for an intuitive user experience.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-brightgreen.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![ElevenLabs](https://img.shields.io/badge/Voice-ElevenLabs-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### 🎭 Advanced Voice Cloning
- **Upload Audio Files**: Clone voices from MP3, WAV, M4A, FLAC, or OGG files
- **Live Recording**: Record your voice directly in the app using your microphone
- **YouTube Integration**: Extract and clone voices from YouTube videos
- **Voice Management**: Switch between cloned voices and manage your voice library

### 🤖 AI-Powered Visual Narration
- **Image Analysis**: Upload photos for detailed AI-generated descriptions
- **Camera Integration**: Capture photos instantly for real-time narration
- **Live Video Streaming**: Real-time video narration with WebRTC support
- **Smart Context**: Adaptive prompting system to avoid repetitive descriptions

### 🔐 Secure API Integration
- **Real-time Validation**: Instant API key verification with detailed error messages
- **Session Management**: Secure handling of API credentials
- **Status Monitoring**: Visual indicators for API connection status
- **Error Handling**: Comprehensive error handling with user-friendly messages

### 🎨 Modern User Interface
- **Clean Design**: Professional Streamlit-based interface
- **Responsive Layout**: Optimized for desktop and mobile devices
- **Intuitive Navigation**: Organized tabs and sections for easy use
- **Visual Feedback**: Clear status indicators and progress messages

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- [OpenAI API Key](https://platform.openai.com/api-keys) (for GPT-4o vision)
- [ElevenLabs API Key](https://www.elevenlabs.io/) (for voice synthesis)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd tts-narrator
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

---

## 🔧 Configuration

### API Keys Setup

You can configure your API keys in two ways:

**Option 1: Environment Variables**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
```

**Option 2: Streamlit Interface**
Enter your API keys directly in the sidebar when running the app. The app will validate them in real-time.

### Supported File Formats

**Audio Files (Voice Cloning)**
- MP3, WAV, M4A, FLAC, OGG
- Recommended: 30+ seconds of clear speech

**Image Files (Narration)**
- JPG, JPEG, PNG
- Any resolution (automatically optimized)

---

## 📖 How to Use

### 1. Configure API Keys
- Enter your OpenAI and ElevenLabs API keys in the sidebar
- Wait for the green "Connected" status indicators

### 2. Clone Your Voice (Optional)
- **Upload Method**: Choose an audio file with your voice
- **Recording Method**: Record directly using your microphone
- **YouTube Method**: Paste a YouTube URL to clone the speaker's voice

### 3. Generate Narrations
- **Static Images**: Upload photos or use your camera
- **Live Video**: Enable real-time video narration with your webcam
- **Voice Selection**: Use your cloned voice or the default ElevenLabs voice

---

## 🏗️ Technical Architecture

### Core Components
- **Frontend**: Streamlit web interface
- **Vision AI**: OpenAI GPT-4o for image understanding
- **Voice AI**: ElevenLabs for speech synthesis and voice cloning
- **Audio Processing**: Pygame for audio playback
- **Video Streaming**: WebRTC for real-time video processing

### Performance Optimizations
- **Async Processing**: Non-blocking audio generation and playback
- **Image Compression**: Optimized JPEG encoding for faster processing
- **Memory Management**: Efficient temporary file handling
- **Concurrent Processing**: Multi-threaded audio and video processing

---

## 📦 Dependencies

### Core Requirements
```
streamlit>=1.46.1
openai>=1.93.0
elevenlabs>=2.6.0
opencv-python>=4.12.0
pygame>=2.6.1
requests>=2.32.4
```

### Optional Features
```
streamlit-webrtc>=0.63.3      # Live video support
streamlit-mic-recorder>=0.0.8  # Voice recording
yt-dlp>=2025.6.30             # YouTube audio extraction
```

---

## 🛠️ Development

### Project Structure
```
tts-narrator/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
└── temp_audio/           # Temporary audio files (auto-created)
```

### Key Functions
- `test_openai_api()` - Validates OpenAI API credentials
- `test_elevenlabs_api()` - Validates ElevenLabs API credentials
- `clone_voice_from_audio()` - Creates voice clones from audio data
- `describe_and_speak()` - Generates descriptions and speech
- `VideoProcessor` - Handles real-time video processing

---

## 🔒 Privacy & Security

- **API Keys**: Stored securely in session state, never logged
- **Audio Files**: Temporary files are automatically cleaned up
- **No Data Storage**: No user data is permanently stored
- **Local Processing**: All processing happens on your machine

---

## 🐛 Troubleshooting

### Common Issues

**API Connection Errors**
- Verify your API keys are correct and active
- Check your internet connection
- Ensure you have sufficient API credits

**Audio Playback Issues**
- Install pygame: `pip install pygame`
- Check your system audio settings
- Try restarting the application

**Video Streaming Problems**
- Install WebRTC: `pip install streamlit-webrtc`
- Allow camera permissions in your browser
- Use a supported browser (Chrome, Firefox)

**Voice Recording Issues**
- Install mic recorder: `pip install streamlit-mic-recorder`
- Allow microphone permissions
- Check your microphone settings

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

<div align="center">
  <p><strong>🎙️ TTS Narrator</strong></p>
  <p>Transform your visual world into spoken stories with AI</p>
</div>
