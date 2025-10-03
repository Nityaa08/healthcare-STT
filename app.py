import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import nltk
import librosa

# Download NLTK data once
nltk.download("stopwords")
nltk.download("punkt")

st.title("Healthcare Audio Keyword Extraction with Mic")

# Load ASR and keyword extraction models once
@st.cache_resource(show_spinner=False)
def load_models():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model.eval()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=embedding_model)
    return processor, asr_model, kw_model

processor, asr_model, kw_model = load_models()

audio_frames = []

def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray(format="flt32")
    audio_frames.append(audio.flatten())
    return frame

webrtc_ctx = webrtc_streamer(
    key="mic",
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if st.button("Stop and Process Audio"):
    if not audio_frames:
        st.warning("No audio recorded yet! Please speak into the mic.")
    else:
        audio_np = np.concatenate(audio_frames)
        orig_sr = 48000  # typical mic audio sample rate
        target_sr = 16000

        # Resample to 16kHz
        audio_16k = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)

        # ASR inference
        input_values = processor(audio_16k, sampling_rate=target_sr, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = asr_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0]).lower()

        st.subheader("Transcription")
        st.write(transcription)

        # Keyword extraction
        keywords = kw_model.extract_keywords(transcription, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=10)

        st.subheader("Extracted Keywords")
        for kw, score in keywords:
            st.write(f"{kw} (score: {score:.2f})")

        # Reset audio buffer for next recording
        audio_frames.clear()
