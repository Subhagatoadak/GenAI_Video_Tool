# 🎥 Streamlit Video Analytics & Transcription Tool

## 📌 Overview
This Streamlit application enables users to upload a video file and extract various insights, including:
- **Video Information**: Duration, frame rate, and resolution
- **Audio Extraction & Transcription**: Uses Whisper AI to transcribe speech
- **Speaker Diarization**: Identifies different speakers using Pyannote
- **Speaker Segmentation**: Matches transcribed text with identified speakers
- **Visual Representation**: Displays a timeline chart of speaker activity

## 🚀 Features
- **Upload & Process Video Files** (`.mp4`, `.avi`, `.mov`, `.mkv`)
- **Automatic Transcription** using OpenAI Whisper
- **Speaker Recognition & Diarization** via Hugging Face Pyannote
- **Speaker Segmentation** by mapping speech segments to speakers
- **Interactive UI** with session persistence to avoid reprocessing on button clicks
- **Speaker Timeline Visualization** using Matplotlib

## 📥 Installation
Clone the repository and install the dependencies:

```sh
# Clone the repository
git clone https://github.com/your-repo/streamlit-video-analytics.git
cd streamlit-video-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🔑 Environment Variables
Create a `.env` file in the project root and add the following keys:

```sh
HUGGINGFACE_API_KEY=your_huggingface_api_key
OPENAI_API_KEY=your_openai_api_key
```

## ▶️ Running the App
Start the Streamlit app:

```sh
streamlit run app.py
```

## 📌 Usage
1. **Upload a video file** via the sidebar
2. **Extract video & audio information**
3. **Transcribe speech** with Whisper AI
4. **Perform speaker diarization** to identify different speakers
5. **View speaker segmentation** with combined text data
6. **Analyze speaker timeline** using a visual chart

## 📊 Technologies Used
- **Streamlit** - For UI development
- **Whisper AI** - For speech-to-text transcription
- **Pyannote** - For speaker diarization
- **MoviePy** - For video processing
- **Matplotlib** - For speaker timeline visualization

## 📝 TODO List
1. **Add VectorDB** to enable semantic search on transcriptions and speaker data.
2. **Break the video into smaller clips** for easier analysis and processing.
3. **Create a highlight video** that summarizes key moments in the uploaded video.
4. **Tone and person face features** that allows to add context.


## 🤝 Contributing
1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License.

---

📩 **For issues or suggestions, create an issue in the repository.**

