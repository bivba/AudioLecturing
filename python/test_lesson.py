import pandas as pd
import datetime
import locale
import re

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

# Copy the improved get_lesson function for testing
def test_get_lesson():
    """Test the improved get_lesson function"""
    
    df = pd.read_csv('schedule.csv', header=None)
    
    def get_lesson(group_column=120):
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
            print(df.iloc[row_indices, column])
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
            is_odd_week = (current_week) % 2 == 1
            
            print(f"Current week: {current_week}, is_odd_week: {is_odd_week}")

            df_time = df[[0, 1]].copy()
            df_time.loc[:, 0] = df_time[0].ffill()
            df_time.loc[:, 1] = df_time[1].ffill()
            
            now = datetime.datetime.now()
            today = now.strftime("%A").capitalize()
            #current_time = now.strftime("%H:%M")
            current_time = "15:20"  # For testing purposes
            current_time_obj = datetime.datetime.strptime(current_time, "%H:%M").time()
            
            print(f"Today: {today}, Current time: {current_time}")
            
            today_sch = df_time[df_time[0] == today].copy()
            if today_sch.empty:
                print(f"No schedule found for {today}")
                return 'Окно'
            
            today_sch['start_time'] = today_sch[1].apply(get_start_time)
            today_sch['end_time'] = today_sch[1].apply(get_end_time)
            today_sch = today_sch.dropna(subset=['start_time', 'end_time'])
            
            current_slots = today_sch[
                (today_sch['end_time'] >= current_time_obj) & 
                (today_sch['start_time'] <= current_time_obj)
            ]
            
            if current_slots.empty:
                print(f"No current lesson at {current_time}")
                return 'Окно'
            
            current_slot = current_slots.tail(1)
            slot_index = current_slot.index[0]
            
            print(f"Found time slot at index: {slot_index}")
            
            if is_odd_week:
                possible_rows = [slot_index - 1, slot_index]
            else:
                possible_rows = [slot_index, slot_index - 1]
            
            lesson = get_lesson_from_rows(possible_rows, group_column)
            
            if lesson:
                print(f"Found lesson: {lesson}")
                return lesson
            else:
                return 'Okno'
                    
        except Exception as e:
            print(f"Error in get_lesson: {e}")
            return 'Окно'
    
    # Test with different group columns
    print("Testing with column 120:")
    lesson = get_lesson(120)
    print(f"Result: {lesson}\n")
    
    print("Testing with column 2 (first group):")
    lesson = get_lesson(2)
    print(f"Result: {lesson}\n")
    
    return lesson

if __name__ == "__main__":
    test_get_lesson()