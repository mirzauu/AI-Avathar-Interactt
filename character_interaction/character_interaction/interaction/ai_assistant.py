import os
from django.conf import settings
import pygame
import speech_recognition as sr
from gtts import gTTS
from google.cloud import texttospeech
from transformers import pipeline
import google.generativeai as genai
import concurrent.futures
import pronouncing  # Library for text-to-phoneme conversion
import re
import subprocess
import json
from pydub import AudioSegment  # Import pydub for audio conversion

# Set the path to the ffmpeg executable
ffmpeg_path = r"C:\Users\alimi\Downloads\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
os.environ["FFMPEG_BINARY"] = ffmpeg_path

class AI_Assistant:
    def __init__(self):
        genai.configure(api_key="AIzaSyAOc5EURc-Xp28JnItSImvT8q5sftnUun0")
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 500,
                "response_mime_type": "text/plain",
            }
        )

        self.full_transcript = [
            {"role": "user", "parts": ["You are a teacher for class of a primary school. Be cheerful "]},
        ]
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        pygame.mixer.init()

        self.emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

    async def process_input(self, transcript):
        self.full_transcript.append({"role": "user", "parts": [transcript]})
        
        chat_session = self.model.start_chat(history=self.full_transcript)
        response = chat_session.send_message(transcript)
        
        response_text = response.text.strip()
        ai_response = self.clean_text(response_text)

        self.full_transcript.append({"role": "model", "parts": [ai_response]})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_emotion = executor.submit(self.detect_emotion, ai_response)
            emotion = future_emotion.result()

        audio_file = self.generate_audio(ai_response)
        wav_file = audio_file.replace('.mp3', '.wav')

        phonemes = self.extract_phonemes_from_text(ai_response)

        # Convert mp3 to wav
        audio = AudioSegment.from_mp3(audio_file)
        audio.export(wav_file, format="wav")

        # Generate lip sync data using Rhubarb Lip Sync
        lip_sync_file_path = os.path.join(settings.MEDIA_ROOT, "response.json")
        rhubarb_exe_path = r"C:\Users\alimi\Downloads\Rhubarb-Lip-Sync-1.13.0-Windows\rhubarb.exe"
        rhubarb_command = f'"{rhubarb_exe_path}" -o "{lip_sync_file_path}" -f json "{wav_file}"'
        subprocess.run(rhubarb_command, shell=True, check=True)

        # Generate the audio URL or path
        audio_url = f"{settings.MEDIA_URL}response.mp3"

        # Print the emotion and phonemes
        print(f"Emotion: {emotion}")
        print("Phonemes:")
        for phoneme in phonemes:
            print(phoneme)

        return ai_response, emotion, audio_url, phonemes, lip_sync_file_path


    def clean_text(self, text):
        # Remove unwanted characters (symbols and emojis)
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove symbols
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove non-ASCII characters (including emojis)
        return cleaned_text.strip()

    def detect_emotion(self, text):
        results = self.emotion_classifier(text)
        emotion = max(results[0], key=lambda x: x['score'])['label']
        return emotion

    def generate_audio(self, text):
        output_file = os.path.join(settings.MEDIA_ROOT, "response.mp3")
        try:
            self.generate_audio_google(text, output_file)
        except Exception as e:
            print(e)
            print("Error with Google TTS, using gTTS as fallback.")
            tts = gTTS(text=text, lang='en')
            tts.save(output_file)
        
        return output_file

    def generate_audio_google(self, text, output_file):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to file {output_file}")

    def extract_phonemes_from_text(self, text):
        phonemes = []
        words = text.split()
        for word in words:
            phones = pronouncing.phones_for_word(word)
            if phones:
                phonemes.append({'word': word, 'phonemes': phones[0]})
            else:
                phonemes.append({'word': word, 'phonemes': []})
        return phonemes

    def play_audio(self, audio_file):
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        os.remove(audio_file)
