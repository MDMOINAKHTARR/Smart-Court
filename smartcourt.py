import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
from transformers import pipeline

# Load Whisper model
print("🔊 Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Whisper ready!")

# Load Summarizer
print("🧠 Loading summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Summarizer ready!")

def record_audio(duration=10, samplerate=16000):
    print(f"\n🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return recording, samplerate

def transcribe_with_whisper(audio, rate):
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        wav.write(temp_file.name, rate, audio)
        print("🔍 Transcribing...")
        result = model.transcribe(temp_file.name)
        return result["text"]

def summarize(text):
    if len(text) < 100:
        return "ℹ Text too short to summarize."
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summary += result[0]['summary_text'] + " "
    return summary.strip()

# === Run the pipeline ===
if _name_ == "_main_":
    audio, rate = record_audio(duration=60)
    transcript = transcribe_with_whisper(audio, rate)

    print("\n📝 Transcript:\n", transcript)

    if transcript.strip():
        summary = summarize(transcript)
        print("\n📄 Summary:\n", summary)