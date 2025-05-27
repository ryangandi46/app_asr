import streamlit as st
import json
import torch
import librosa
import os
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    AutoModelForCTC, Wav2Vec2Processor,
    pipeline
)
from jiwer import wer

# === Load models ===
@st.cache_resource
def load_models():
    base_path = "asr_models"  # lokal
    # whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(base_path + "/whisper_model").to("cpu")
    # whisper_proc = AutoProcessor.from_pretrained(base_path + "/whisper_processor")
    # whisper_pipe = pipeline("automatic-speech-recognition", model=whisper_model,
    #                         tokenizer=whisper_proc.tokenizer,
    #                         feature_extractor=whisper_proc.feature_extractor,
    #                         device=-1)
    
    # Load Whisper model dan processor
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(base_path + "/whisper_model").to("cpu")
    whisper_proc = AutoProcessor.from_pretrained(base_path + "/whisper_processor")

    w2v2_model = AutoModelForCTC.from_pretrained(base_path + "/wav2vec2_model").to("cpu")
    w2v2_proc = Wav2Vec2Processor.from_pretrained(base_path + "/wav2vec2_processor")
    w2v2_pipe = pipeline("automatic-speech-recognition", model=w2v2_model,
                         tokenizer=w2v2_proc.tokenizer,
                         feature_extractor=w2v2_proc.feature_extractor,
                         device=-1)

    # return whisper_pipe, w2v2_pipe
    return (whisper_model, whisper_proc), w2v2_pipe

# === Load dataset ===
@st.cache_data
def load_dataset(path="./cleaned_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# whisper_pipe, w2v2_pipe = load_models()
(whisper_model, whisper_proc), w2v2_pipe = load_models()
dataset = load_dataset()

# === UI ===
st.title("📊 Evaluasi ASR: Whisper vs Wav2Vec2")
index = st.slider("Pilih Sampel", min_value=0, max_value=len(dataset)-1, value=0)
sample = dataset[index]

st.audio(sample["path"], format="audio/wav")
st.write(f"**Teks Referensi:** {sample['text']}")

# Load audio
audio, sr = librosa.load(sample["path"], sr=16000)

# === Whisper ===
# st.subheader("🔊 Whisper")
# try:
#     whisper_out = whisper_pipe({"raw": audio, "sampling_rate": 16000})
#     whisper_text = whisper_out["text"]
#     st.text_area("Transkripsi Whisper", whisper_text, height=100)
#     whisper_wer = wer(sample["text"].lower(), whisper_text.lower())
#     st.write(f"WER (Whisper): **{whisper_wer:.3f}**")
# except Exception as e:
#     st.error(f"Gagal Whisper: {e}")

# === Whisper ===
st.subheader("🔊 Whisper")
try:
    inputs = whisper_proc(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = whisper_model.generate(inputs["input_features"])
    whisper_text = whisper_proc.batch_decode(outputs, skip_special_tokens=True)[0]
    
    st.text_area("Transkripsi Whisper", whisper_text, height=100)
    whisper_wer = wer(sample["text"].lower(), whisper_text.lower())
    st.write(f"WER (Whisper): **{whisper_wer:.3f}**")
except Exception as e:
    st.error(f"Gagal Whisper: {e}")


# === Wav2Vec2 ===
st.subheader("🔊 Wav2Vec2")
try:
    w2v2_out = w2v2_pipe(audio)
    w2v2_text = w2v2_out["text"]
    st.text_area("Transkripsi Wav2Vec2", w2v2_text, height=100)
    w2v2_wer = wer(sample["text"].lower(), w2v2_text.lower())
    st.write(f"WER (Wav2Vec2): **{w2v2_wer:.3f}**")
except Exception as e:
    st.error(f"Gagal Wav2Vec2: {e}")
