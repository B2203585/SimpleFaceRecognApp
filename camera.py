import os
import time
from collections import Counter, deque
import cv2
import numpy as np
import streamlit as st

# Import từ file mới ml_models
from ml_models import load_or_train_model, extract_hog

def run_data_collection(placeholder, person_name, max_img, dataset_dir="dataset"):
    # (Đoạn này giữ nguyên y hệt code cũ của bạn)
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    start_wait = time.time()
    skip_waiting = False

    while st.session_state.run_cam:
        ret, frame = cap.read()
        if not ret: break
        remaining = int(5 - (time.time() - start_wait))
        display = frame.copy()
        if remaining > 0 and not skip_waiting:
            cv2.putText(display, "STATUS: Waiting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(display, f"Taking photo in: {remaining}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Press 'S' to skip", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            placeholder.image(display, channels="BGR", width=640)
            if cv2.waitKey(1) & 0xFF == ord("s"):
                skip_waiting = True
            continue
        break

    count = 0
    while st.session_state.run_cam and count < max_img:
        ret, frame = cap.read()
        if not ret: break
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) > 0:
            status_txt = "STATUS: Face Detected"
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            face_roi = frame[y : y + h, x : x + w]
            cv2.imwrite(os.path.join(person_dir, f"img_{count}.jpg"), cv2.resize(face_roi, (160, 160)))
            count += 1
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"Saved: {count}/{max_img}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            status_txt = "STATUS: Searching..."
        cv2.putText(display, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        placeholder.image(display, channels="BGR", width=640)
        if count >= max_img:
            st.session_state.run_cam = False
            st.success(f"Đã thu thập xong {max_img} ảnh!")
            break
        cv2.waitKey(1)
    cap.release()
    st.rerun()


def run_live_recognition(placeholder, model_type):
    # Nạp Model thông minh (Load từ ổ cứng hoặc Train mới)
    with st.spinner(f"Đang kiểm tra và nạp Model {model_type}..."):
        clf, label_encoder = load_or_train_model(model_type=model_type)

    if clf is None:
        st.error("Dữ liệu trống! Hãy thu thập ảnh trước.")
        st.session_state.run_cam = False
        st.rerun()
        return

    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    buffer = deque(maxlen=15)
    
    # Ngưỡng cho từng Model
    KNN_DIST_THRESHOLD = 6.0 
    SVM_PROB_THRESHOLD = 0.65 # Phải chắc chắn > 65% mới nhận diện

    while st.session_state.run_cam:
        ret, frame = cap.read()
        if not ret: break

        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_full, 1.2, 8, minSize=(100, 100))

        face_detected = len(faces) > 0
        for (x, y, w, h) in faces:
            start_t = time.perf_counter()
            feat = extract_hog(gray_full[y : y + h, x : x + w])[None, :]
            
            # Xử lý theo từng Model
            if model_type == "KNN":
                dists, indices = clf.kneighbors(feat, n_neighbors=3)
                avg_dist = sum(dists[0]) / 3
                if avg_dist < KNN_DIST_THRESHOLD:
                    pred_id = clf.predict(feat)[0]
                    label = label_encoder.inverse_transform([pred_id])[0]
                    color = (255, 0, 0)
                    metric_txt = f"{avg_dist:.2f}"
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    metric_txt = f"{avg_dist:.2f}"
            else: # Thuật toán SVM
                proba = clf.predict_proba(feat)[0]
                max_prob = np.max(proba)
                if max_prob > SVM_PROB_THRESHOLD:
                    pred_id = np.argmax(proba)
                    label = label_encoder.inverse_transform([pred_id])[0]
                    color = (255, 0, 0)
                    metric_txt = f"{max_prob*100:.0f}%"
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    metric_txt = f"{max_prob*100:.0f}%"

            buffer.append(label)
            voted_label = Counter(buffer).most_common(1)[0][0]

            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, f"{voted_label} ({metric_txt})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        status_msg = "STATUS: Face Detected" if face_detected else "STATUS: Searching..."
        cv2.putText(display, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        placeholder.image(display, channels="BGR", width=640)

    cap.release()
    st.rerun()