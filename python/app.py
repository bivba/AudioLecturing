import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
import re
import threading
from concurrent.futures import TimeoutError as FutureTimeout

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
from VideoRecording import VideoRecorder
from vector_db import get_vector_db
from utils import *


merge_locks = {}
merge_locks_lock = threading.Lock()

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")
executor = ThreadPoolExecutor(max_workers=2)
app = Flask(__name__)
CORS(app)


print('loading model...')
bot = TelegramBot()
summary = Summariser(model_name='gemini-flash-latest')
print('model loaded')

vector_db = get_vector_db()

video_recorder = None
video_lock = threading.Lock()

telegram_loop = None

def start_polling():
    global telegram_loop
    loop = asyncio.new_event_loop()
    telegram_loop = loop
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot.poll_messages())
    except Exception as e:
        print(f"Polling thread error: {e}")
    finally:
        loop.close()


polling_thread = threading.Thread(target=start_polling, daemon=True)
polling_thread.start()
print("Started Telegram polling")

df = pd.read_csv('schedule.csv', header=None)

GROUP_COLUMN = 120


@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    global video_recorder
    try:
        with video_lock:
            if video_recorder is not None and getattr(video_recorder, 'recording', False):
                print("Screen recording already in progress")
                return jsonify({"status": "already_recording"}), 200
            video_recorder = VideoRecorder('screenshots')
            video_recorder.record()
            print("Started screen recording via /start_recording")
        return jsonify({"status": "started"}), 200
    except Exception as e:
        print(f"Error starting screen recording: {e}")
        return jsonify({"error": str(e)}), 500

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

    try:
        executor.submit(process_chunk, session_id, abs_path, chunk_index)

    except Exception as e:
        return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

    return jsonify({"status": "accepted", "session_id": session_id, "chunk_index": chunk_index})

@app.route('/merge', methods=['POST'])
def merge_summaries():
    lesson_name = get_lesson(df, GROUP_COLUMN)
    global video_recorder
    session_id = request.form.get('session_id')
    audio_file = request.files.get('audio')

    with merge_locks_lock:
        if session_id not in merge_locks:
            merge_locks[session_id] = threading.Lock()
        lock = merge_locks[session_id]

    acquired = lock.acquire(blocking=False)
    if not acquired:
        print(f"Merge already in progress for session {session_id}")
        return jsonify({"error": "Merge already in progress for this session"}), 429
    
    try:
        if audio_file:
            filename = secure_filename(audio_file.filename or f"final_{session_id}.webm")
            os.makedirs("uploads", exist_ok=True)
            audio_path = os.path.join("uploads", filename)
            audio_file.save(audio_path)

            fixed_path = fix_webm(audio_path)
            transcript = summary.transcribe_audio(fixed_path)

            out_dir = "transcripts"
            os.makedirs(out_dir, exist_ok=True)
            outpath = os.path.join(out_dir, f"{session_id}_final.md")

            with open(outpath, "w", encoding="utf-8") as f:
                f.write(transcript)
        else:
            transcript = ""

        full_transcript = ""
        transcript_dir = "transcripts"
        if os.path.exists(transcript_dir):
            transcript_files = sorted(os.listdir(transcript_dir))
            for filename in transcript_files:
                with open(os.path.join(transcript_dir, filename), "r", encoding="utf-8") as f:
                    full_transcript += f.read() + " "

        screenshots = []
        with video_lock:
            if video_recorder is not None:
                try:
                    video_recorder.stop()
                    screenshots = video_recorder.get_screenshots()
                    video_recorder.release()
                    print(f"Collected {len(screenshots)} screenshots")
                except Exception as e:
                    print(f"Error collecting screenshots: {e}")
                finally:
                    video_recorder = None
            else:
                print("No active screen recorder; skipping screenshots.")

        if transcript is None:
            transcript = ""
        combined_text = transcript + " " + full_transcript
        result = summary.summarize_text(combined_text, screenshots)

        vector_db.add_document(result, metadata={'lesson': lesson_name, 'date': str(datetime.datetime.now())})

        now = datetime.datetime.now()
        current_date = now.strftime("%d.%m")

        if telegram_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                bot.send_message(result, f"{lesson_name}_{current_date}.md"),
                #bot.send_message(result, f'Английский_видео.md'), 
                telegram_loop
            )
            try:
                future.result(timeout=10)
            except FutureTimeout:
                print("Telegram send timed out scheduling")
            except Exception as e:
                print(f"Error scheduling Telegram send: {e}")
        else:
            print("No telegram loop available; skipping send.")
        
        cleanup_session(session_id)

        return jsonify({"status": "merged"})
    
    finally:
        lock.release()
        with merge_locks_lock:
            try:
                del merge_locks[session_id]
            except KeyError:
                pass

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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)