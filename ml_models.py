import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def extract_hog(gray):
    gray_res = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=(96, 96),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(gray_res).flatten().astype(np.float32)

def get_dataset_state(dataset_dir):
    """Đếm tổng số lượng ảnh trong dataset để biết dữ liệu có thay đổi không"""
    count = 0
    if not os.path.exists(dataset_dir): return 0
    for root, dirs, files in os.walk(dataset_dir):
        count += len([f for f in files if f.lower().endswith(('.jpg', '.png'))])
    return count

def load_or_train_model(dataset_dir="dataset", model_type="KNN"):
    model_path = f"{model_type.lower()}_model.pkl"
    le_path = "label_encoder.pkl"
    state_path = "dataset_state.txt"

    current_state = get_dataset_state(dataset_dir)
    if current_state == 0:
        return None, None

    # Kiểm tra xem có cần Train lại không
    need_train = True
    if os.path.exists(model_path) and os.path.exists(le_path) and os.path.exists(state_path):
        with open(state_path, 'r') as f:
            saved_state = int(f.read().strip())
        if saved_state == current_state:
            need_train = False

    if not need_train:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(le_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder

    # NẾU CẦN TRAIN LẠI
    x_data, y_data = [], []
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png"))]
        for file_name in files:
            img = cv2.imread(os.path.join(class_dir, file_name), 0)
            if img is not None:
                x_data.append(extract_hog(img))
                y_data.append(cls)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    # Khởi tạo Model theo lựa chọn
    if model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    elif model_type == "SVM":
        model = SVC(kernel='poly', C=1.0, probability=True) 

    model.fit(x_data, y_encoded)

    # Lưu Model, Label Encoder và Trạng thái dữ liệu xuống file
    with open(model_path, 'wb') as f: pickle.dump(model, f)
    with open(le_path, 'wb') as f: pickle.dump(label_encoder, f)
    with open(state_path, 'w') as f: f.write(str(current_state))

    return model, label_encoder