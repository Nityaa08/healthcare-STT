import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from io import BytesIO
from streamlit_audio_recorder import audio_recorder

# Load models once at startup
@st.cache_resource
def load_models():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model.eval()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model=embedding_model)
    return processor, asr_model, kw_model

processor, asr_model, kw_model = load_models()

st.title("Healthcare Audio Keyword Extractor 🎤")

st.write("Record your voice, then we'll transcribe and extract keywords.")

# Record audio from mic
audio_bytes = audio_recorder()

if audio_bytes is not None:
    st.audio(audio_bytes, format="audio/wav")

    # Load audio with librosa from bytes
    audio_np, sr = librosa.load(BytesIO(audio_bytes), sr=16000)

    # Run ASR
    input_values = processor(audio_np, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower()

    st.subheader("Transcription")
    st.write(transcription)

    # Extract keywords
    keywords = kw_model.extract_keywords(
        transcription, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10
    )

    st.subheader("Extracted Keywords")
    for kw, score in keywords:
        st.write(f"- {kw} (score: {score:.3f})")
else:
    st.info("Click the record button to start recording your voice.")
