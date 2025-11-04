# 🧠 TTS Narrator: Image & Video Description with Voice Cloning

A multimodal AI-powered narrator app that **describes images or live video scenes** using **GPT-4o**, and speaks them aloud using **ElevenLabs voice cloning**. Built with **Streamlit**, it supports **voice cloning from uploaded files, mic recordings, or YouTube**, and narrates both static and real-time content.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![ElevenLabs](https://img.shields.io/badge/Voice-ElevenLabs-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 Features

### 🖼️ Image & Live Video Narration

- Upload an image or use your **webcam** to get AI-generated spoken descriptions
- Uses **GPT-4o vision** for intelligent scene understanding
- Async streaming for real-time narration

### 🧬 Voice Cloning & Text-to-Speech (TTS)

- Clone your voice from:
  - 🔼 Uploaded audio files
  - 🎙️ Mic recordings (via `streamlit-mic-recorder`)
  - 📺 YouTube video links
- Uses **ElevenLabs API** for lifelike speech
- Custom voice stored for future use in narration

### 📸 WebRTC Video Support

- Real-time video narration using `streamlit-webrtc`
- Optimized JPEG compression for performance
- Adaptive prompting to avoid repetitive descriptions

### 🧠 Intelligent Prompting

- Context-aware prompt cycling to avoid repetition
- Recent response filtering using text similarity
- Silences avoided with auto-trigger after a threshold

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [ElevenLabs API Key](https://www.elevenlabs.io/)

### 🧰 Installation

```bash
# 1. Clone the Repository
git clone https://github.com/akaTalhakhan/VISION_bASED_VOICE_ELEVEN_LABS.git
cd VISION_bASED_VOICE_ELEVEN_LABS

# 2. Create a Virtual Environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the App
streamlit run main.py

# 5. Open in Browser
http://localhost:8501
```

---

## 📁 Project Structure

```
VISION_bASED_VOICE_ELEVEN_LABS/
├── .env                # 🔐 API keys for OpenAI and ElevenLabs
├── .venv/              # 🧪 Virtual environment
├── main.py             # 🚀 Streamlit app with narration logic
├── requirements.txt    # 📦 Required Python packages
└── README.md           # 📘 You're reading it!
```

---

## 🔐 API Key Setup

Create a `.env` file in the root directory with:

```
OPENAI_API_KEY=your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
```

Or enter them via the Streamlit sidebar at runtime.

---

## 📣 Voice Models

Supports all ElevenLabs default voices and custom cloned voices. The default fallback is:

- `o1GYQOsPeFjZRhnXfhdg` — ElevenLabs Default

Cloned voices are cached during your session and deletable from the UI.

---

## 🧪 Dependencies

Some optional modules used:

- `streamlit-mic-recorder` – mic input
- `streamlit-webrtc` – live webcam video support
- `pygame` – audio playback
- `yt-dlp` – for voice cloning from YouTube

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
