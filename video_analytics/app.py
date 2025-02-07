import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import os
import ffmpeg
import pandas as pd
import matplotlib.pyplot as plt
from pyannote.audio import Pipeline
import openai
from streamlit_extras.metric_cards import style_metric_cards
import streamlit_scrollable_textbox as stx
from streamlit_timeline import timeline
from streamlit_text_rating.st_text_rater import st_text_rater
from streamlit_player import st_player
from dotenv import load_dotenv

st.set_page_config(page_title="Video Analytics & Transcription", page_icon="🎥", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f4f4;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .subheader {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        .fixed-video {
            width: 640px !important;
            height: 360px !important;
        }
    </style>
""", unsafe_allow_html=True)


# Load environment variables
load_dotenv()

# Retrieve API keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_API_KEY)

def convert_audio_to_wav(input_path, output_path):
    """ Convert audio to 16kHz WAV format """
    ffmpeg.input(input_path).output(output_path, acodec="pcm_s16le", ar="16000").run(overwrite_output=True)
    
def clean_segments(segments):
    """ Extracts text and timestamps from segments """
    cleaned_data = []
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        cleaned_data.append([start_time, end_time, text])
    return pd.DataFrame(cleaned_data, columns=["Start Time", "End Time", "Text"])

def match_segments_to_diarization(diarization_df, segment_df):
    """ Matches segment text to diarization based on overlapping time ranges """
    matched_data = []
    for _, dia_row in diarization_df.iterrows():
        matching_texts = segment_df[(segment_df["Start Time"] >= dia_row["Start Time"]) & 
                                    (segment_df["End Time"] <= dia_row["End Time"])]
        text = " ".join(matching_texts["Text"]) if not matching_texts.empty else ""
        matched_data.append([dia_row["Start Time"], dia_row["End Time"], dia_row["Speaker"], text])
    return pd.DataFrame(matched_data, columns=["Start Time", "End Time", "Speaker", "Text"])

st.sidebar.markdown("<div class='main-title'>🎥 Upload Video</div>", unsafe_allow_html=True)
st.sidebar.write("Upload a video file to extract audio, transcribe speech, analyze speakers, and generate insights.")


uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with st.spinner("Processing video..."):
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        video_clip = VideoFileClip(temp_video_path)
        video_duration = round(video_clip.duration, 2)
        video_fps = round(video_clip.fps,0)
        video_resolution = f"{video_clip.size[0]}x{video_clip.size[1]}"
        
        audio_clip = video_clip.audio
        temp_audio_path = "temp_audio.wav"
        audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
        
        temp_audio_16k_path = "temp_audio_16k.wav"
        convert_audio_to_wav(temp_audio_path, temp_audio_16k_path)
        
    st.video(temp_video_path, format='video/mp4', start_time=0)
    
    with st.spinner("Transcribing audio..."):
        result = model.transcribe(temp_audio_16k_path, word_timestamps=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        
        st.markdown("<div class='subheader'>📊 Video Information</div>", unsafe_allow_html=True)
        ol1, ol2, ol3 = st.columns(3)
        ol1.metric("Duration", f"{video_duration} sec")
        ol2.metric("Frame Rate", f"{video_fps} FPS")
        ol3.metric("Resolution", video_resolution)
        st.audio(temp_audio_16k_path)

    
    with col2:
        st.markdown("<div class='subheader'>📜 Transcription</div>", unsafe_allow_html=True) 
        transcript_text = result["text"]
        stx.scrollableTextbox(transcript_text)
      
    
    st.markdown("<div class='subheader'>📜 Segments</div>", unsafe_allow_html=True)    
    segments = result["segments"]
    segment_df = clean_segments(segments)
    st.dataframe(segment_df,use_container_width=True)        
        #stx.scrollableTextbox(str(segments))
    
    with st.spinner("Performing speaker diarization..."):
        diarization = pipeline(temp_audio_16k_path)
    
    
    diarization_data = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_data.append([turn.start, turn.end, speaker])

    col5, col6 = st.columns([2, 2])
    
    diarization_df = pd.DataFrame(diarization_data, columns=["Start Time", "End Time", "Speaker"])
    combined_df = match_segments_to_diarization(diarization_df, segment_df)
    with col5:
      st.markdown("<div class='subheader'>🔖 Speaker Analysis</div>", unsafe_allow_html=True)
      st.dataframe(combined_df,use_container_width=True)
      
    with col6:
      st.markdown("<div class='subheader'>📊 Diarization Chart</div>", unsafe_allow_html=True)
      fig, ax = plt.subplots()
      for _, row in diarization_df.iterrows():
          ax.barh(row["Speaker"], row["End Time"] - row["Start Time"], left=row["Start Time"], height=0.4)
      plt.xlabel("Time (seconds)")
      plt.ylabel("Speakers")
      plt.title("Speaker Timeline")
      st.pyplot(fig)
     

    
    st.markdown("<div class='subheader'>📌 Insights</div>", unsafe_allow_html=True)
    with st.spinner("Generating Key Insights..."):
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract the agenda, key points, sentiment, positive and negative sections, of the given text."},
                {"role": "user", "content": transcript_text}
            ]
                                                    )
        stx.scrollableTextbox(response.choices[0].message.content.strip(),height="5px")
    
    
    
    
    os.remove(temp_video_path)
    os.remove(temp_audio_path)
    os.remove(temp_audio_16k_path)
