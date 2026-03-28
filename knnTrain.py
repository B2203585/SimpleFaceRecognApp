import os

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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


def auto_train(dataset_dir="dataset"):
    if not os.path.exists(dataset_dir):
        return None, None

    x_data, y_data = [], []
    classes = [
        directory
        for directory in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, directory))
    ]

    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        files = [
            file_name
            for file_name in os.listdir(class_dir)
            if file_name.lower().endswith((".jpg", ".png"))
        ]

        for file_name in files:
            img = cv2.imread(os.path.join(class_dir, file_name), 0)
            if img is not None:
                x_data.append(extract_hog(img))
                y_data.append(cls)

    if not x_data:
        return None, None

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    model.fit(x_data, y_encoded)
    return model, label_encoder
