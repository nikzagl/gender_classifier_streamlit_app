from transformers import AutoModelForAudioClassification, TrainingArguments
from transformers import AutoFeatureExtractor
import streamlit as st
import librosa 
import torch
audio_value = st.audio_input("Record a voice message")
model = AutoModelForAudioClassification.from_pretrained("nikzagl/wav2vec-finetuned-on-gender-classification")

if audio_value:
    st.audio(audio_value)
    audio, sr = librosa.load(audio_value, sr = 16000)
    with torch.no_grad():
        feature_extractor = AutoFeatureExtractor.from_pretrained("nikzagl/wav2vec-finetuned-on-gender-classification")
        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_class_ids = torch.argmax(logits).item()
        if predicted_class_ids == 1:
            st.write("Female")
        else:
            st.write("Male")
            