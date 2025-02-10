# 🎥 Streamlit Video Analytics & Editing Tool

## 📌 Overview
This Streamlit application enables users to upload a video file and extract various insights, including:
- **Video Information**: Duration, frame rate, and resolution
- **Audio Extraction & Transcription**: Uses Whisper AI to transcribe speech
- **Speaker Diarization**: Identifies different speakers using Pyannote
- **Speaker Segmentation**: Matches transcribed text with identified speakers
- **Semantic Search**: Retrieves relevant segments using FAISS and OpenAI embeddings
- **Video Editing**: Adds text overlays and allows merging extracted video clips

## 🚀 Features
- **Upload & Process Video Files** (`.mp4`, `.avi`, `.mov`, `.mkv`)
- **Automatic Transcription** using OpenAI Whisper
- **Speaker Recognition & Diarization** via Hugging Face Pyannote
- **Speaker Segmentation** by mapping speech segments to speakers
- **Semantic Search** using FAISS vector database and OpenAI embeddings
- **Clip Extraction** from search results
- **Video Editing** with text overlays
- **Concatenate Extracted Clips** into a final video

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
### **Video Page**
1. **Upload a video file** via the sidebar
2. **Extract video & audio information**
3. **Transcribe speech** with Whisper AI
4. **Perform speaker diarization** to identify different speakers
5. **View speaker segmentation** with combined text data
6. **Analyze speaker timeline** using a visual chart

### **Semantic Search Page**
1. **Enter a search term** to find relevant video segments
2. **View top 10 matching segments**
3. **Extract and display clips from the video**
4. **Download individual extracted clips**
5. **Combine all extracted clips into a single video**

### **Video Editing Page**
1. **Select a video clip**
2. **Enter text for overlay**
3. **Adjust position, font size, and color**
4. **Apply the overlay and download the edited video**

## 📊 Technologies Used
- **Streamlit** - For UI development
- **Whisper AI** - For speech-to-text transcription
- **Pyannote** - For speaker diarization
- **MoviePy** - For video processing and editing
- **FAISS** - For semantic search indexing
- **OpenAI Embeddings** - For efficient text retrieval

## 📝 TODO List
1. **Add More Video Editing Features** (e.g., trimming, filters, overlays)
2. **Improve Semantic Search** with additional NLP techniques
3. **Enhance UI/UX** for better user interaction
4. **Integrate Cloud Storage** for saving processed clips

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

