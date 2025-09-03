import os
import asyncio
import whisper
import pandas as pd
import datetime
import locale
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

from TelegramSend import TelegramBot
from Summary import Summariser
locale.setlocale(locale.LC_TIME, "Russian_Russia.1251")

app = Flask(__name__)
CORS(app)

print('loading model...')
model = whisper.load_model("base")
print('model loaded')

bot = TelegramBot()
print(os.getenv("GOOGLE_API_KEY"))
summariser = Summariser()
df = pd.read_csv('schedule.csv', header=None)
df_time = df[[0, 1]]
df_time[0] = df[0].fillna(method='ffill')
df_time[1] = df[1].fillna(method='ffill')
now = datetime.datetime.now()
today = now.strftime("%A").capitalize()
current_time = now.strftime("%H:%M")
today_sch = df_time[df_time[0] == today]
today_time = today_sch[today_sch[1] <= current_time].tail(1)
lesson = df.loc[today_time.index][120]
if lesson.isna().sum() == 1:
    lesson = 'Окно'


@app.route('/summarize_audio', methods=['POST'])
def summarize_audio():
    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename or "temp.webm")
    os.makedirs("uploads", exist_ok=True)
    audio_path = os.path.join("uploads", filename)
    audio_file.save(audio_path)
    abs_path = os.path.abspath(audio_path)
    print("Saved file to:", abs_path)

    # Transcribe the audio file
    try:
        print('transcribing audio...')
        result = model.transcribe(abs_path, best_of=3, beam_size=5, fp16=False)
        asr_out = [result['text']]
        print('transcription complete')

        if result and result['text']:
            print('generating summary')
            summary = summariser.summarize_text(result['text'])
            print('summary generated')
            with open(lesson + '.md', "w", encoding="utf-8") as f:
                f.write(summary)
            print('sending message to Telegram')
            with open(lesson + '.md', 'rb') as f:
                asyncio.run(bot.send_message(f))

    except Exception as e:
        return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

    finally:
        #os.remove(audio_path)
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(lesson + ".md"): os.remove(lesson + ".md")
        print("Cleaned up temporary files.")

    return jsonify({"text": result})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)