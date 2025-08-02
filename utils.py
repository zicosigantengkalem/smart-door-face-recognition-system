import os
import face_recognition
from datetime import datetime
from playsound import playsound
import pandas as pd

def load_known_faces(dataset_path="dataset/authorized"):
    known_encodings = []
    known_names = []
    if not os.path.isdir(dataset_path):
        return known_encodings, known_names
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
            except Exception:
                continue
    return known_encodings, known_names

def play_alarm(alarm_path="alarm.mp3"):
    try:
        playsound(alarm_path, block=False)
    except Exception as e:
        print("Alarm error:", e)

def log_event(name, status, actor="system"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now},{actor},{name},{status}\n"
    with open("log.csv", "a") as f:
        f.write(line)

def load_log():
    if os.path.exists("log.csv"):
        df = pd.read_csv("log.csv", names=["Timestamp", "Actor", "DetectedName", "Status"])
        return df
    return None
