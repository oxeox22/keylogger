import sqlite3
import logging
import threading
from queue import Queue
import time
from pynput import keyboard
import subprocess

# 로깅 및 데이터베이스 경로 설정
log_dir = r"C:\Users\jh052\Desktop\SWUFORCE\제4회 교내연합세미나\keylogger\seol_slow_typing_data.db"
log_file_path = log_dir + "keyboard.txt"
db_file_path = 'typing_data.db'  # 데이터베이스 파일 경로

# SQLite 데이터베이스에 연결
conn = sqlite3.connect(db_file_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, uri=True)

# 파일 로깅 설정
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='["%(asctime)s", %(message)s]')
logging.basicConfig(filename=log_dir + "\\keyboard.txt", level=logging.DEBUG, format='["%(asctime)s", %(message)s]')

key_press_time = 0  # 키가 눌린 시간을 저장하는 변수
key_down = False #키 다운 상태 여부
prev_key_release_time = 0 #이전 키가 눌린 시간을 저장하는 변수

# 큐 초기화
queue = Queue()

# 데이터 삽입 함수
def insert_interval(table_name, key, timestamp, duration, duration2=None, duration3=None):
    # with 문을 사용하여 데이터베이스 연결 관리
    with sqlite3.connect(db_file_path) as conn:
        c = conn.cursor()

        c.execute(f'CREATE TABLE IF NOT EXISTS {table_name} (key TEXT, timestamp REAL, duration REAL, duration2 REAL, duration3 REAL)')

        c.execute(f'INSERT INTO {table_name} (key, timestamp, duration, duration2, duration3) VALUES (?, ?, ?, ?, ?)', 
                  (key, timestamp, duration, duration2, duration3))

        conn.commit()

# 큐 처리 함수
def process_queue():
    while True:
        data = queue.get()
        if data is None:
            break

        if len(data) == 4:  # 데이터가 4개인 경우
            table_name, key, timestamp, duration = data
            insert_interval(table_name, key, timestamp, duration)
        elif len(data) == 6: # 데이터가 6개인 경우
            table_name, key, timestamp, duration, duration2, duration3 = data
            insert_interval(table_name, key, timestamp, duration, duration2, duration3)
            

#백스페이스 횟수 변수 초기화
backspace_count = 0
total_keys_pressed = 0

# 데이터 삽입 스레드 생성 및 시작
thread = threading.Thread(target=process_queue)
thread.start()

# 키 입력 이벤트 처리 함수
def on_press(key):
    global key_press_time, backspace_count, total_keys_pressed, key_down
    try:
        char_key = key.char.lower()
        key_press_time = time.time()
        total_keys_pressed += 1
        key_down = True  # 키 다운 상태
        
    except AttributeError:
        if key == keyboard.Key.space:
            char_key = 'space'
            key_press_time = time.time()
            total_keys_pressed += 1
            key_down = True  # 키 다운 상태
        elif key == keyboard.Key.backspace:
            char_key = 'backspace'
            key_press_time = time.time()
            total_keys_pressed += 1
            backspace_count += 1
            error_rate = (backspace_count/total_keys_pressed)*100
            insert_backspace_count('backspace', total_keys_pressed, backspace_count, error_rate)
            key_down = True  # 키 다운 상태
        elif key == keyboard.Key.shift:
            char_key = 'shift'
            key_press_time = time.time()
            total_keys_pressed += 1
            key_down = True
        elif key == keyboard.Key.enter:  # enter 누를 때
            subprocess.Popen(['python', 'yesorno.py'])  # svdd로 넘어가게
            raise KeyboardInterrupt 


def on_release(key):
    global key_press_time, prev_key_release_time, key_down
    release_time = None  # Initialize release_time with None
    try:
        char_key = key.char
        release_time = time.time()
        duration = release_time - key_press_time
        duration2 = key_press_time - prev_key_release_time if prev_key_release_time and key_down else 0
        duration3 = duration + duration2 if duration and duration2 else 0
        
        queue.put(('typing_data', char_key, key_press_time, duration, duration2, duration3))
        prev_key_release_time = release_time
        key_down = False #키 업 상태

    except AttributeError:
        if key == keyboard.Key.space:
            char_key = 'space'
            release_time = time.time()
            duration = release_time - key_press_time
            duration2 = key_press_time - prev_key_release_time if prev_key_release_time and key_down else 0
            duration3 = duration + duration2 if duration and duration2 else 0
            
            queue.put(('typing_data', char_key, key_press_time, duration, duration2, duration3))
            key_down = False
        elif key == keyboard.Key.backspace:
            char_key = 'backspace'
            release_time = time.time()
            duration = release_time - key_press_time
            duration2 = key_press_time - prev_key_release_time if prev_key_release_time and key_down else 0
            duration3 = duration + duration2 if duration and duration2 else 0
            
            queue.put(('typing_data', char_key, key_press_time, duration, duration2, duration3))
            key_down = False
        elif key == keyboard.Key.shift:
            char_key = 'shift'
            release_time = time.time()
            duration = release_time - key_press_time
            duration2 = key_press_time - prev_key_release_time if prev_key_release_time and key_down else 0
            duration3 = duration + duration2 if duration and duration2 else 0
            
            queue.put(('typing_data', char_key, key_press_time, duration, duration2, duration3))
            key_down = False
        elif key == keyboard.Key.enter:
            char_key = 'enter'
            release_time = time.time()
            duration = release_time - key_press_time
            duration2 = key_press_time - prev_key_release_time if prev_key_release_time and key_down else 0
            duration3 = duration + duration2 if duration and duration2 else 0
            
            queue.put(('typing_data', char_key, key_press_time, duration, duration2, duration3))
            key_down = False

        prev_key_release_time = release_time

#백스페이스 횟수 데이터 삽입 함수
def insert_backspace_count(table_name, all_count, back_count, error_rate):
    with sqlite3.connect(db_file_path) as conn:
        c = conn.cursor()

        c.execute(f'CREATE TABLE IF NOT EXISTS {table_name} (all_count INTEGER, back_count INTEGER, error_rate INTEGER)')

        c.execute(f'INSERT INTO {table_name} (all_count, back_count, error_rate) VALUES (?, ?, ?)', (all_count, back_count, error_rate))

        conn.commit()

# 키보드 리스너 생성 및 시작
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# 큐에 None을 추가하여 데이터 삽입 스레드 종료
queue.put(None)
# 데이터 삽입 스레드가 종료될 때까지 대기
thread.join()

# 연결 종료
conn.close()
