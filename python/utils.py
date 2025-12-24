import os
import asyncio
import datetime
import re
import pandas as pd
import shutil
import subprocess
import locale


def get_lesson(df, group_column=120):

    def get_start_time(time):
        if pd.isna(time):
            return None
        match = re.match(r"^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*$", str(time))
        if match:
            start_time_str = match.group(1)
            return datetime.datetime.strptime(start_time_str, "%H:%M").time()
        else:
            return None

    def get_end_time(time):
        if pd.isna(time):
            return None
        match = re.match(r"^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*$", str(time))
        if match:
            end_time_str = match.group(2)
            return datetime.datetime.strptime(end_time_str, "%H:%M").time()
        else:
            return None

    def get_lesson_from_rows(row_indices, column):
        lessons = []
        
        for idx in row_indices:
            if idx < len(df) and column < len(df.columns):
                lesson = df.iloc[idx, column]
                if pd.notna(lesson) and str(lesson).strip():
                    lessons.append(str(lesson).strip())

        if lessons:
            return lessons[0]
        return None

    try:
        current_week = datetime.datetime.now().isocalendar()[1]
        is_odd_week = (current_week + 1) % 2 == 1
        
        print(f"Current week: {current_week}, is_odd_week: {is_odd_week}")

        # Prepare time data
        df_time = df[[0, 1]].copy()
        df_time.loc[:, 0] = df_time[0].ffill()
        df_time.loc[:, 1] = df_time[1].ffill()

        now = datetime.datetime.now()
        today = now.strftime("%A").capitalize()
        current_time = now.strftime("%H:%M")
        current_time_obj = datetime.datetime.strptime(current_time, "%H:%M").time()

        today_sch = df_time[df_time[0] == today].copy()
        if today_sch.empty:
            print(f"No schedule found for {today}")
            return 'Окно'
        
        # Add time columns
        today_sch['start_time'] = today_sch[1].apply(get_start_time)
        today_sch['end_time'] = today_sch[1].apply(get_end_time)
        
        # Filter out rows with invalid times
        today_sch = today_sch.dropna(subset=['start_time', 'end_time'])
        
        # Find current time slot
        current_slots = today_sch[
            (today_sch['end_time'] >= current_time_obj) &
            (today_sch['start_time'] <= current_time_obj)
        ]
        
        if current_slots.empty:
            print(f"No current lesson at {current_time}")
            return 'Окно'
        
        # Get the time slot (use the last matching slot if multiple)
        current_slot = current_slots.tail(1)
        slot_index = current_slot.index[0]
        
        print(f"Found time slot at index: {slot_index}")
        
        # For odd/even week handling, we typically check consecutive rows
        # The assumption is that odd week lessons are in one row, even week in the next
        # Adjust this logic based on your specific schedule structure
        
        if is_odd_week:
            # Check current row first, then next row as fallback
            possible_rows = [slot_index - 1, slot_index]
        else:
            # Check next row first, then current row as fallback
            possible_rows = [slot_index, slot_index - 1]

        # Try to get lesson from the determined rows
        lesson = get_lesson_from_rows(possible_rows, group_column)
        
        if lesson:
            print(f"Found lesson: {lesson}")
            return lesson
        else:
            return 'Окно'
                
    except Exception as e:
        print(f"Error in get_lesson: {e}")
        return 'Окно'


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
        session_dir = f'screenshots/'
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            print(f"Cleaned up screenshots directory: {session_dir}")
    except Exception as e:
        print(f"Error during cleanup : {e}")
