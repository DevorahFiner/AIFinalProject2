import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import os

# Initialize emotion analysis pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

st.title("Speech-to-Text & Emotion Analysis")
st.write("Use your voice to input speech, and we'll transcribe it and analyze the emotions behind the words!")

# Speech recognition
recognizer = sr.Recognizer()

with st.form(key="speech_form"):
    st.write("Press 'Record' and say something to analyze its emotions.")
    audio_record = st.form_submit_button("Record")

    if audio_record:
        with sr.Microphone() as source:
            st.write("Recording... Speak now!")
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                st.write(f"Transcribed Text: {text}")
                
                # Perform emotion analysis on the transcribed text
                emotion_scores = emotion_classifier(text)
                st.write("Emotion Scores:")
                for score in emotion_scores[0]:
                    st.write(f"{score['label']}: {score['score']:.2f}")

                # Optional: Generate audio response
                tts = gTTS(text, lang="en")
                tts.save("speech.mp3")
                st.audio("speech.mp3")
                
            except Exception as e:
                st.write(f"Error: {e}")
