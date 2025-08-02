from ultralytics import YOLO

def load_thief_model(path="models/best.pt"):
    model = YOLO(path)
    return model

def is_thief_yolo(model, face_img, conf_threshold=0.5):
    results = model(face_img, verbose=False)[0]
    thief_detected = False
    best_conf = 0.0
    for box in results.boxes:
        conf = float(box.conf.cpu().numpy())
        if conf >= conf_threshold:
            thief_detected = True
            if conf > best_conf:
                best_conf = conf
    return thief_detected, best_conf
