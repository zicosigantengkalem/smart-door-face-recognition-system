import os
import face_recognition
import numpy as np

DATASET_DIR = "dataset_public"
SAVE_FILE = "known_faces.npy"

def preprocess_and_save_faces(dataset_dir, save_file):
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset folder '{dataset_dir}' not found. Please create this folder and add face images.")
        return

    print(f"Starting face preprocessing from '{dataset_dir}'...")
    
    for name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {name}")
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            encoding = encodings[0]
                            known_face_encodings.append(encoding)
                            known_face_names.append(name)
                            print(f"  Successfully processed: {image_path}")
                        else:
                            print(f"  Warning: No faces detected in image: {image_path}")
                    except Exception as e:
                        print(f"  Error while processing {image_path}: {e}")

    if len(known_face_encodings) > 0:
        final_encodings_array = np.vstack(known_face_encodings).astype(np.float64)
        final_names_array = np.array(known_face_names, dtype=object)

        data_to_save = {
            'encodings': final_encodings_array,
            'names': final_names_array
        }
        np.save(save_file, data_to_save, allow_pickle=True)
        print(f"Preprocessing complete. {len(known_face_encodings)} faces successfully saved to '{save_file}'.")
    else:
        print("No faces were processed. File was not saved.")

if __name__ == "__main__":
    preprocess_and_save_faces(DATASET_DIR, SAVE_FILE)

    


