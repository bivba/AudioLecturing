import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess

import pandas as pd
import datetime
import locale
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid
load_dotenv()

from TelegramSend import TelegramBot
from Summary import Summariser
locale.setlocale(locale.LC_TIME, "Russian_Russia.1251")

executor = ThreadPoolExecutor(max_workers=2)

app = Flask(__name__)
CORS(app)

bot = TelegramBot()

print('loading model...')
summary = Summariser()
print('model loaded')

df = pd.read_csv('schedule.csv', header=None)


@app.route('/summarize_audio', methods=['POST'])
def summarize_audio():
    session_id = request.form.get("session_id", str(uuid.uuid4()))
    chunk_index = request.form.get("chunk_index", "0")
    audio_file = request.files['audio']
    print('received chunk')
    filename = secure_filename(audio_file.filename or "temp.webm")
    os.makedirs("uploads", exist_ok=True)
    audio_path = os.path.join("uploads", filename)
    audio_file.save(audio_path)
    abs_path = os.path.abspath(audio_path)
    print("Saved file to:", abs_path)

    # Transcribe the audio file
    try:
        executor.submit(process_chunk, session_id, abs_path, chunk_index)

    except Exception as e:
        return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

    return jsonify({"status": "accepted", "session_id": session_id, "chunk_index": chunk_index})

@app.route('/merge', methods=['POST'])
def merge_summaries():
    session_id = request.form.get('session_id')
    audio_file = request.files.get('audio')

    if audio_file:
        # Save last uploaded file
        filename = secure_filename(audio_file.filename or f"final_{session_id}.webm")
        os.makedirs("uploads", exist_ok=True)
        audio_path = os.path.join("uploads", filename)
        audio_file.save(audio_path)

        # Process last chunk (blocking, since it's final)
        transcript = summary.transcribe_audio(audio_path)
    else:
        transcript = ""

    # Collect existing transcript files
    full_transcript = ""
    transcript_dir = "transcripts"
    if os.path.exists(transcript_dir):
        transcript_files = sorted(os.listdir(transcript_dir))
        for filename in transcript_files:
            with open(os.path.join(transcript_dir, filename), "r", encoding="utf-8") as f:
                full_transcript += f.read() + " "

    # Append final transcript if any
    combined_text = transcript + " " + full_transcript
    result = summary.summarize_text(combined_text)

    try:
        asyncio.run(bot.send_message(result, f"{get_lesson()}.md"))
    except Exception as e:
        print(f"Error sending message to Telegram: {e}")
    
    cleanup_session(session_id)

    return jsonify({"status": "merged"})

def get_lesson():
    df_time = df[[0, 1]]
    df_time[0] = df[0].fillna(method='ffill')
    df_time[1] = df[1].fillna(method='ffill')
    now = datetime.datetime.now()
    today = now.strftime("%A").capitalize()
    current_time = now.strftime("%H:%M")
    today_sch = df_time[df_time[0] == today]
    today_time = today_sch[today_sch[1] <= current_time].tail(1)
    lesson = df.loc[today_time.index][120].values[0]
    if pd.isna(lesson):
        lesson = 'Окно'

    return lesson

def process_chunk(session_id, filepath, chunk_index):
    try:
        print(f'[{session_id}] processing chunk {chunk_index}')
        out_dir = "transcripts"
        os.makedirs(out_dir, exist_ok=True)
        fixed_path = fix_webm(filepath)
        result = summary.transcribe_audio(fixed_path)
        print(f'[{session_id}] finished processing chunk {chunk_index}')
        outpath = os.path.join(out_dir, f"{session_id}_{chunk_index}.md")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(result)
        return result
    except Exception as e:
        print(f'[{session_id}] error processing chunk {chunk_index}: {e}')
        return None

def fix_webm(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_fixed.wav"
    try:

        command = [
            "ffmpeg",
            "-i", input_path,      # Input file (let FFmpeg figure out the container)
            "-y",                  # Overwrite output file if it exists
            "-ar", "16000",        # Resample to 16kHz for Whisper
            "-ac", "1",            # Convert to mono
            "-c:a", "pcm_s16le",   # Specify the standard WAV audio codec
            output_path            # The output file
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True, text=True
        )
        os.remove(input_path)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error fixing webm with FFmpeg: {e.stderr}")
        return input_path
    except Exception as e:
        print(f"An unexpected error occurred in fix_webm: {e}")
        return input_path

def cleanup_session(session_id):
    try:
        session_dir = 'uploads'
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            print(f"Cleaned up session directory: {session_dir}")
        session_dir = 'transcripts'
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            print(f"Cleaned up session directory: {session_dir}")
    except Exception as e:
        print(f"Error during cleanup for session {session_id}: {e}")



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)