import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, ImageClip
import whisper
import os
import ffmpeg
import pandas as pd
import matplotlib.pyplot as plt
from pyannote.audio import Pipeline
import openai
import streamlit_scrollable_textbox as stx
from streamlit_timeline import timeline
from dotenv import load_dotenv
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(page_title="Video Analytics", page_icon="🎥", layout="wide")
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
client = openai.OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_API_KEY)

def convert_audio_to_wav(input_path, output_path):
    """ Convert audio to 16kHz WAV format """
    ffmpeg.input(input_path).output(output_path, acodec="pcm_s16le", ar="16000").run(overwrite_output=True)

def cut_video_clip(input_video, start_time, end_time, output_path):
    """ Cuts a segment of the video between start_time and end_time """
    if input_video is None or not os.path.exists(input_video):
        raise ValueError("Invalid video file path")
    
    video = VideoFileClip(input_video)
    start_time = max(0, min(start_time, video.duration))
    end_time = max(start_time, min(end_time, video.duration))
    
    if start_time >= end_time:
        raise ValueError("Start time must be less than end time")
    
    subclip = video.subclip(start_time, end_time)
    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def add_text_to_video(input_video, output_video, text, position=(10, 10), fontsize=50, color='white'):
    """ Adds text overlay to a video clip without requiring ImageMagick """
    if input_video is None or not os.path.exists(input_video):
        raise ValueError("Invalid video file path")

    video = VideoFileClip(input_video)

    # Create a transparent image for text overlay
    txt_img = Image.new('RGBA', (video.w, video.h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_img)

    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    txt_clip = ImageClip(np.array(txt_img)).set_duration(video.duration)

    # Overlay the text clip on the original video
    result = CompositeVideoClip([video, txt_clip])
    result.write_videofile(output_video, codec="libx264", audio_codec="aac")
    
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

def encode_text(text):
    """ Encodes text using the model """
    response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)
    return response.data[0].embedding

st.sidebar.markdown("<div class='main-title'>🎥 Upload Video</div>", unsafe_allow_html=True)
st.sidebar.write("Upload a video file to extract audio, transcribe speech, analyze speakers, and generate insights.Search keywords, create short clips, and edit videos.")
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
page = st.sidebar.radio("Go to", ["Video Page", "Semantic Search Page","Video Editing Page"])



# Session State Initialization
if "video_clip" not in st.session_state:
    st.session_state.video_clip = None
if "audio_clip" not in st.session_state:
    st.session_state.audio_clip = None
if "video_duration" not in st.session_state:
    st.session_state.video_duration = None
if "video_fps" not in st.session_state:
    st.session_state.video_fps = None
if "video_resolution" not in st.session_state:
    st.session_state.video_resolution = None    
if "response" not in st.session_state:
    st.session_state.response = None
if "search_query" not in st.session_state:
    st.session_state.search_query = None
if "result" not in st.session_state:
    st.session_state.result = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "diarization_df" not in st.session_state:
    st.session_state.diarization_df = None
if "combined_df" not in st.session_state:
    st.session_state.combined_df = None
if "segment_df" not in st.session_state:
    st.session_state.segment_df = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "clips" not in st.session_state:
    st.session_state.clips = []
if "edited_clips" not in st.session_state:
    st.session_state.edited_clips = []
if "video_path" not in st.session_state:
    st.session_state.video_path = None




if uploaded_file is not None:
    if page == "Video Page":
        st.markdown("<div class='main-title'>🎥 Video Analytics & Transcription</div>", unsafe_allow_html=True)
        with st.spinner("Processing video..."):
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.video_path=temp_video_path
            st.session_state.video_clip = VideoFileClip(temp_video_path)
            st.session_state.video_duration = round(st.session_state.video_clip.duration, 2)
            st.session_state.video_fps = round(st.session_state.video_clip.fps,0)
            st.session_state.video_resolution = f"{st.session_state.video_clip.size[0]}x{st.session_state.video_clip.size[1]}"
            
            st.session_state.audio_clip = st.session_state.video_clip.audio
            temp_audio_path = "temp_audio.wav"
            st.session_state.audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
            
            temp_audio_16k_path = "temp_audio_16k.wav"
            convert_audio_to_wav(temp_audio_path, temp_audio_16k_path)
            
        st.video(temp_video_path, format='video/mp4', start_time=0)
        
        with st.spinner("Transcribing audio..."):
            st.session_state.result = model.transcribe(temp_audio_16k_path, word_timestamps=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            st.markdown("<div class='subheader'>📊 Video Information</div>", unsafe_allow_html=True)
            ol1, ol2, ol3 = st.columns(3)
            ol1.metric("Duration", f"{st.session_state.video_duration} sec")
            ol2.metric("Frame Rate", f"{st.session_state.video_fps} FPS")
            ol3.metric("Resolution", st.session_state.video_resolution)
            st.audio(temp_audio_16k_path)

        
        with col2:
            st.markdown("<div class='subheader'>📜 Transcription</div>", unsafe_allow_html=True) 
            st.session_state.transcript_text = st.session_state.result["text"]
            stx.scrollableTextbox(st.session_state.transcript_text)
        
        
        st.markdown("<div class='subheader'>📜 Segments</div>", unsafe_allow_html=True)    
        segments = st.session_state.result["segments"]
        segment_df = clean_segments(segments)
        st.dataframe(segment_df,use_container_width=True)
        
        with st.spinner("Loading to Database..."):
            chunk_embeddings = []
            chunk_metadata = []
            
            for _, row in segment_df.iterrows():
                embedding = encode_text(row["Text"])
                chunk_embeddings.append(np.array(embedding).reshape(1, -1))
                chunk_metadata.append((row["Start Time"], row["End Time"], row["Text"]))
            
            
            embeddings_array = np.vstack(chunk_embeddings)
            index = faiss.IndexFlatL2(embeddings_array.shape[1])
            index.add(embeddings_array)
        
            st.session_state.faiss_index = index
            st.session_state.metadata = chunk_metadata
            st.session_state.segment_df = segment_df
        st.success("Successfully added to the index!")        
            #stx.scrollableTextbox(str(segments))
        
        with st.spinner("Performing speaker diarization..."):
            diarization = pipeline(temp_audio_16k_path)
        
        
        diarization_data = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_data.append([turn.start, turn.end, speaker])

        col5, col6 = st.columns([2, 2])
        
        st.session_state.diarization_df = pd.DataFrame(diarization_data, columns=["Start Time", "End Time", "Speaker"])
        st.session_state.combined_df = match_segments_to_diarization(st.session_state.diarization_df, segment_df)
        with col5:
            st.markdown("<div class='subheader'>🔖 Speaker Analysis</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.combined_df,use_container_width=True)
            
        with col6:
            st.markdown("<div class='subheader'>📊 Diarization Chart</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            for _, row in st.session_state.diarization_df.iterrows():
                ax.barh(row["Speaker"], row["End Time"] - row["Start Time"], left=row["Start Time"], height=0.4)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Speakers")
            plt.title("Speaker Timeline")
            st.pyplot(fig)
            

        
        st.markdown("<div class='subheader'>📌 Insights</div>", unsafe_allow_html=True)
        with st.spinner("Generating Key Insights..."):
            
            st.session_state.response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract the agenda, key points, sentiment, positive and negative sections, of the given text."},
                    {"role": "user", "content": st.session_state.transcript_text}
                ]
                                                        )
            stx.scrollableTextbox(st.session_state.response.choices[0].message.content.strip(),height="5px")
        
        
        

if page == "Semantic Search Page":
    st.title("🔍 Semantic Search in Transcription")
    st.session_state.search_query = st.text_input("Enter a search term:")
    
    if st.session_state.search_query and st.session_state.faiss_index is not None:
        query_embedding = np.array([encode_text(st.session_state.search_query)]).astype("float32")
        
        if query_embedding.shape[1] != st.session_state.faiss_index.d:
            st.error("Query embedding dimension does not match FAISS index.")
        else:
            distances, indices = st.session_state.faiss_index.search(query_embedding, 10)
            results = [st.session_state.metadata[i] for i in indices[0] if i < len(st.session_state.metadata)]
            
            if results:
                st.markdown("### Top 10 Matches")
                search_df = pd.DataFrame(results, columns=["Start Time", "End Time", "Text"])
                st.dataframe(search_df, use_container_width=True)

                for i, (start_time, end_time, text) in enumerate(results):
                    output_clip_path = f"clip_{i}.mp4"
                    st.write("video")
                    cut_video_clip(st.session_state.video_path, start_time, end_time, output_clip_path)
                    st.session_state.clips.append(output_clip_path)
                    
                    st.markdown(f"#### Clip {i+1}")
                    with st.expander(text, expanded=False):
                        st.video(output_clip_path)
                        st.download_button("Download Clip", open(output_clip_path, "rb").read(), file_name=output_clip_path)
                    
                if st.button("Combine Extracted Clips"):
                    combined_clip = concatenate_videoclips([VideoFileClip(clip) for clip in st.session_state.clips])
                    combined_clip_path = "combined_output.mp4"
                    combined_clip.write_videofile(combined_clip_path, codec="libx264", audio_codec="aac")
                    st.markdown("### Combined Video")
                    st.video(combined_clip_path)
                    st.download_button("Download Combined Video", open(combined_clip_path, "rb").read(), file_name=combined_clip_path)
            else:
                st.warning("No relevant results found.")

if page == "Video Editing Page":
    st.title("🎬 Video Editing Tool")
    if st.session_state.clips:
        clip_choice = st.selectbox("Select a Clip to Edit", st.session_state.clips)
        text_input = st.text_input("Enter text to overlay on video:")
        position_x = st.slider("X Position", 0, 500, 10)
        position_y = st.slider("Y Position", 0, 500, 10)
        fontsize = st.slider("Font Size", 20, 100, 50)
        color = st.color_picker("Text Color", "#ffffff")

        if st.button("Apply Text Overlay"):
            output_edited_clip = f"edited_{clip_choice}"
            try:
                add_text_to_video(clip_choice, output_edited_clip, text_input, position=(position_x, position_y), fontsize=fontsize, color=color)
                st.session_state.edited_clips.append(output_edited_clip)
                st.video(output_edited_clip)
                st.download_button("Download Edited Clip", open(output_edited_clip, "rb").read(), file_name=output_edited_clip)
            except ValueError as e:
                st.error(f"Error processing video: {e}")
    else:
        st.warning("No extracted clips found. Please perform a semantic search first.")

