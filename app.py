import streamlit as st
import cv2
import numpy as np
import os
import face_recognition
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import io
import base64

st.set_page_config(page_title="Smart Door Face Recognition", layout="wide")
st.title("🚪 Smart Door Face Recognition System By zicosigantengkalem")

PUBLIC_DATASET_DIR = "dataset_public"
SAVE_FILE = "known_faces.npy"
LOG_FILE = "log.csv"
THIEF_MODEL_PATH = "models/best.pt"
ALARM_FILE = "alarm.mp3"

ADMIN_USER = "zicosideveloper"
ADMIN_PASS = "zicosangpembuatsystem"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_success_message" not in st.session_state:
    st.session_state.login_success_message = False
if "new_face_photos" not in st.session_state:
    st.session_state.new_face_photos = []
if "face_tolerance" not in st.session_state:
    st.session_state.face_tolerance = 0.45
if "thief_threshold" not in st.session_state:
    st.session_state.thief_threshold = 0.9
if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False
if "alarm_playing" not in st.session_state:
    st.session_state.alarm_playing = False
if "alarm_stopped_manually" not in st.session_state:
    st.session_state.alarm_stopped_manually = False

@st.cache_data(show_spinner="Loading thief detection model...")
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"YOLO model not found at: {THIEF_MODEL_PATH}")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

thief_model = load_yolo_model(THIEF_MODEL_PATH)

@st.cache_data(show_spinner="Loading face data...")
def load_processed_known_faces(save_file=SAVE_FILE):
    if not os.path.exists(save_file):
        st.warning("Known faces database not found. No faces can be recognized.")
        return [], []
    try:
        data = np.load(save_file, allow_pickle=True).item()
        return data.get("encodings", []), data.get("names", [])
    except Exception as e:
        st.error(f"Failed to load known faces database: {e}.")
        return [], []

def preprocess_and_store_faces(dataset_dir=PUBLIC_DATASET_DIR, save_file=SAVE_FILE):
    encodings = []
    names = []
    if not os.path.isdir(dataset_dir):
        st.error(f"Dataset folder not found at: {dataset_dir}")
        return False, 0

    person_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not person_folders:
        st.warning("Dataset folder is empty. No faces will be processed.")
        return False, 0
    
    progress_bar = st.progress(0, text="Starting process...")
    all_images = []
    for person_name in person_folders:
        person_folder = os.path.join(dataset_dir, person_name)
        for img_name in os.listdir(person_folder):
            all_images.append((person_name, img_name, os.path.join(person_folder, img_name)))

    total_images = len(all_images)
    processed_images = 0

    for person_name, img_name, img_path in all_images:
        processed_images += 1
        progress_text = f"Processing: {person_name}/{img_name}"
        progress_bar.progress(processed_images / total_images, text=progress_text)
        try:
            image = face_recognition.load_image_file(img_path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(person_name)
            else:
                st.warning(f"No faces detected in image: {img_path}")
        except Exception as e:
            st.warning(f"Failed to process {img_path}: {e}")
            continue
    
    progress_bar.empty()
    if not encodings:
        st.error("No faces were successfully detected from the dataset images.")
        return False, 0

    final_encodings_array = np.vstack(encodings).astype(np.float64)
    final_names_array = np.array(names, dtype=object)

    data_to_save = {
        'encodings': final_encodings_array,
        'names': final_names_array
    }

    np.save(save_file, data_to_save, allow_pickle=True)
    st.cache_data.clear()
    return True, len(names)

def log_event(name, status, actor="System"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{now},{actor},{name},{status}\n")

alarm_html = ""
if os.path.exists(ALARM_FILE):
    try:
        audio_bytes = open(ALARM_FILE, 'rb').read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        alarm_html = f"""
            <audio id="alarm-audio" autoplay>
              <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
    except Exception as e:
        st.error(f"Failed to prepare alarm audio: {e}")
else:
    st.error(f"Alarm file not found at: {ALARM_FILE}")

known_face_encodings, known_face_names = load_processed_known_faces()

menu = st.sidebar.radio("Select Menu:", ["1. Face Detection", "2. Developer Options"])

if menu == "1. Face Detection":
    st.header("👁️ Face Detection")
    st.info("Position your face in front of the camera or upload a photo to start detection.")

    detection_source = st.radio("Select Image Source:", ("Webcam", "Upload Photo"))
    
    image_file_buffer = None
    if detection_source == "Webcam":
        st.subheader("From Webcam")
        image_file_buffer = st.camera_input("Take a photo from the camera")
    elif detection_source == "Upload Photo":
        st.subheader("From an Uploaded Photo")
        image_file_buffer = st.file_uploader("Select an image from your device", type=["jpg", "jpeg", "png"])
    
    if image_file_buffer is None and st.session_state.alarm_active:
        st.session_state.alarm_active = False
        st.session_state.alarm_playing = False
        st.session_state.alarm_stopped_manually = True
        st.markdown("<script>var audio = document.getElementById('alarm-audio'); if (audio) audio.pause();</script>", unsafe_allow_html=True)
    
    face_recognition_tolerance = st.session_state.face_tolerance
    thief_detection_threshold = st.session_state.thief_threshold

    if image_file_buffer is not None:
        st.session_state.alarm_stopped_manually = False

        with st.spinner("Analyzing image..."):
            bytes_data = image_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            processed_image = image.copy()
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            detection_results = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                result = {
                    "name": "Unknown",
                    "confidence": 0.0,
                    "is_known": False,
                    "is_thief": False,
                    "box": (top, right, bottom, left)
                }
                
                if len(known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=face_recognition_tolerance)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        result["name"] = known_face_names[best_match_index]
                        result["confidence"] = confidence
                        result["is_known"] = True
                    else:
                        if thief_model:
                            face_img = image[top:bottom, left:right]
                            results_yolo = thief_model(face_img, verbose=False)[0]
                            if results_yolo.boxes:
                                best_thief_conf = float(results_yolo.boxes[0].conf[0])
                                if best_thief_conf >= thief_detection_threshold:
                                    result["name"] = "Thief"
                                    result["confidence"] = best_thief_conf
                                    result["is_thief"] = True
                
                detection_results.append(result)

            final_result = None
            
            known_faces = [res for res in detection_results if res["is_known"]]
            if known_faces:
                final_result = max(known_faces, key=lambda x: x["confidence"])
            else:
                thief_faces = [res for res in detection_results if res["is_thief"]]
                if thief_faces:
                    final_result = max(thief_faces, key=lambda x: x["confidence"])
                else:
                    unknown_faces = [res for res in detection_results if not res["is_known"] and not res["is_thief"]]
                    if unknown_faces:
                        final_result = max(unknown_faces, key=lambda x: x.get("confidence", 0))

            st.subheader("Detection Result")
            
            if final_result:
                name = final_result["name"]
                confidence = final_result["confidence"]
                top, right, bottom, left = final_result["box"]
                
                label = ""
                color = (0, 255, 255)
                
                if final_result["is_known"]:
                    label = f"SAFE: {name} ({confidence:.2f})"
                    color = (0, 255, 0)
                    log_event(name, "UNLOCKED_SAFE")
                    st.success(f"✅ Door Unlocked! Face recognized as **{name}**.")
                    st.session_state.alarm_active = False
                    st.session_state.alarm_playing = False
                elif final_result["is_thief"]:
                    label = f"THIEF ({confidence:.2f})"
                    color = (0, 0, 255)
                    log_event("Thief", "LOCKED_THIEF_DETECTED")
                    st.error("🚨 DANGER! Thief detected. Door remains locked.")
                    if not st.session_state.alarm_stopped_manually:
                        st.session_state.alarm_active = True
                        st.session_state.alarm_playing = True
                else:
                    label = "Unknown Face"
                    color = (0, 255, 255)
                    log_event("Unknown", "LOCKED_UNKNOWN")
                    st.warning("❓ Unknown face. Door remains locked.")
                    st.session_state.alarm_active = False
                    st.session_state.alarm_playing = False

                cv2.rectangle(processed_image, (left, top), (right, bottom), color, 2)
                cv2.putText(processed_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                st.image(processed_image, channels="BGR", caption="Processed Image.")
            else:
                st.warning("No face detected in the image.")
                st.session_state.alarm_active = False
                st.session_state.alarm_playing = False

    st.markdown("---")
    if st.session_state.alarm_active and st.session_state.alarm_playing:
        st.markdown(alarm_html, unsafe_allow_html=True)
        st.session_state.alarm_playing = False
    
    if st.session_state.alarm_active:
        if st.button("Stop Alarm", key="bottom_stop_alarm"):
            st.session_state.alarm_active = False
            st.session_state.alarm_stopped_manually = True
            st.success("Alarm has been stopped.")
            st.markdown("<script>var audio = document.getElementById('alarm-audio'); if (audio) audio.pause();</script>", unsafe_allow_html=True)


elif menu == "2. Developer Options":
    st.header("⚙️ Developer Options")

    if not st.session_state.authenticated:
        st.write("You must log in as an administrator to access this menu.")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")
            if submitted:
                if username == ADMIN_USER and password == ADMIN_PASS:
                    st.session_state.authenticated = True
                    st.session_state.login_success_message = True
                    st.rerun()
                else:
                    st.error("Invalid username or password!")
        
        with st.expander("Show Developer Credentials"):
            st.write(f"**Username:** `{ADMIN_USER}`")
            st.write(f"**Password:** `{ADMIN_PASS}`")

    if st.session_state.authenticated:
        if st.session_state.login_success_message:
            st.success("Login successful!")
            st.session_state.login_success_message = False

        dev_menu = st.radio("Select Developer Menu:", ["Add Face to Database", "View Detection Log", "System Settings", "Rebuild Face Database"])

        if dev_menu == "Add Face to Database":
            st.subheader("👤 Add Face to Database")
            st.info(f"Note: This feature will add photos to the local face database at `{PUBLIC_DATASET_DIR}`.")
            person_name = st.text_input("Enter Full Name (no spaces, e.g., JohnDoe):")
            
            photo_source = st.radio("Select Photo Source:", ["Webcam", "Upload Photos"])
            
            if photo_source == "Webcam":
                st.write("Take at least 4 photos from different angles.")
                cam_photo = st.camera_input("Take Face Photo", key="add_cam")
                if cam_photo and st.button("Save Photo"):
                    photo_bytes = cam_photo.getvalue()
                    img_np = np.frombuffer(photo_bytes, np.uint8)
                    img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    face_detected = len(face_recognition.face_locations(img_cv2)) > 0
                    st.session_state.new_face_photos.append((photo_bytes, face_detected))
                    if face_detected:
                        st.success("Face detected in this photo! Take more photos.")
                    else:
                        st.warning("No face detected. Please try again.")
            
            elif photo_source == "Upload Photos":
                st.write("Upload at least 4 photos from your device.")
                uploaded_files = st.file_uploader("Select face photo files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
                
                if uploaded_files:
                    st.session_state.new_face_photos = []
                    for file in uploaded_files:
                        photo_bytes = file.getvalue()
                        img_np = np.frombuffer(photo_bytes, np.uint8)
                        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        face_detected = len(face_recognition.face_locations(img_cv2)) > 0
                        st.session_state.new_face_photos.append((photo_bytes, face_detected))

            if st.session_state.new_face_photos:
                if not all(isinstance(p, tuple) and len(p) == 2 for p in st.session_state.new_face_photos):
                    st.session_state.new_face_photos = []
                    st.warning("Photo data from previous session was incompatible and has been reset.")
                    st.rerun()

                st.markdown("---")
                face_detected_photos = [p for p in st.session_state.new_face_photos if p[1]]
                st.subheader(f"Photo Previews ({len(face_detected_photos)}/4 Faces Detected)")
                
                cols = st.columns(4)
                for i, (photo_data, face_detected) in enumerate(st.session_state.new_face_photos):
                    with cols[i % 4]:
                        st.image(photo_data, use_column_width=True)
                        if face_detected:
                            st.success("✅ Face detected.")
                        else:
                            st.error("❌ No face detected.")
                        if st.button("Remove", key=f"remove_photo_{i}"):
                            del st.session_state.new_face_photos[i]
                            st.rerun()
                if st.button("Reset Photos"):
                    st.session_state.new_face_photos = []
                    st.rerun()
            
            st.markdown("---")
            
            face_detected_photos = [p for p in st.session_state.new_face_photos if p[1]]
            is_disabled = len(face_detected_photos) < 4 or not person_name
            
            if is_disabled:
                if not person_name and len(face_detected_photos) < 4:
                    st.warning(f"Please provide a name and add at least 4 photos ({len(face_detected_photos)}/4) with a face detected to update.")
                elif not person_name:
                    st.warning("Please provide a name to update.")
                else:
                    st.warning(f"Please add at least 4 photos ({len(face_detected_photos)}/4) with a face detected to update.")

            if st.button("Update", disabled=is_disabled):
                with st.spinner("Updating face database..."):
                    person_dir = os.path.join(PUBLIC_DATASET_DIR, person_name)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    processed_count = 0
                    for photo_data, face_detected in st.session_state.new_face_photos:
                        if face_detected:
                            cv2_img = cv2.imdecode(np.frombuffer(photo_data, np.uint8), cv2.IMREAD_COLOR)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            img_path = os.path.join(person_dir, f"{timestamp}.jpg")
                            cv2.imwrite(img_path, cv2_img)
                            processed_count += 1
                        else:
                            st.warning("Skipping one photo because no face was detected.")
                    
                    if processed_count > 0:
                        st.success(f"{processed_count} photos for '{person_name}' saved successfully. Running pre-processing...")
                        success, count = preprocess_and_store_faces()
                        if success:
                            st.success(f"Database rebuild complete! {count} face samples processed. Please push the updated 'known_faces.npy' file to GitHub for the public version to work.")
                            st.balloons()
                            st.session_state.new_face_photos = []
                        else:
                            st.error("Pre-processing failed. Check for errors above.")
                    else:
                        st.error("No faces were successfully saved from the provided photos.")
            
        elif dev_menu == "View Detection Log":
            st.subheader("📜 Detection Activity Log")
            if os.path.exists(LOG_FILE):
                try:
                    log_df = pd.read_csv(LOG_FILE, names=["Timestamp", "Actor", "Detected Name", "Status"])
                    
                    log_df = log_df[["Timestamp", "Detected Name", "Status"]]
                    log_df.rename(columns={"Detected Name": "Name"}, inplace=True)
                    
                    status_map = {
                        "UNLOCKED_SAFE": "Door Unlocked (Allowed)",
                        "LOCKED_THIEF_DETECTED": "Door Locked (Thief Detected)",
                        "LOCKED_UNKNOWN": "Door Locked (Unknown Face)"
                    }
                    log_df['Status'] = log_df['Status'].map(status_map)
                    
                    st.dataframe(log_df.sort_values(by="Timestamp", ascending=True), use_container_width=True)
                    
                    if st.button("Clear All Logs"):
                        os.remove(LOG_FILE)
                        st.success("Log file successfully deleted.")
                        st.rerun()
                except pd.errors.EmptyDataError:
                    st.info("Log file is empty.")
                except Exception as e:
                    st.error(f"Failed to read log file: {e}")
            else:
                st.info("No activity has been logged yet.")
        
        elif dev_menu == "System Settings":
            st.subheader("🛠️ System Settings")
            st.write("Adjust the sensitivity of the detection models.")
            st.session_state.face_tolerance = st.slider(
                "Face Recognition Tolerance",
                0.3, 0.7, st.session_state.face_tolerance, 0.05,
                help="A lower value means stricter face recognition."
            )
            st.session_state.thief_threshold = st.slider(
                "Thief Detection Threshold",
                0.3, 0.9, st.session_state.thief_threshold, 0.05,
                help="The minimum confidence level to be flagged as a thief."
            )
        
        elif dev_menu == "Rebuild Face Database":
            st.subheader("🔁 Rebuild Face Database")
            st.warning("This will rescan all existing photos and rebuild the face database.")
            st.info("This is useful if you have manually added or renamed folders in the dataset directory.")
            
            if st.button("Rebuild Database Now"):
                st.session_state.alarm_stopped_manually = False
                st.session_state.alarm_active = False
                st.session_state.alarm_playing = False

                with st.spinner("Rebuilding face database..."):
                    success, count = preprocess_and_store_faces()
                    if success:
                        st.success(f"Database rebuild complete! {count} face samples processed. Please push the updated 'known_faces.npy' file to GitHub for the public version to work.")
                        st.balloons()
                    else:
                        st.error("Database rebuild failed. Check for errors above.")

        if st.sidebar.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.login_success_message = False
            st.rerun()
