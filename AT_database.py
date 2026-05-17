# AT_database.py
import os
import cv2
import numpy as np
from preprocessing import standardize_face  # <--- Import your new tool


def load_att_dataset(dataset_path):
    X_train, y_train, X_test, y_test = [], [], [], []

    for subject_id in range(1, 41):
        folder_path = os.path.join(dataset_path, f"s{subject_id}")
        for image_num in range(1, 11):
            image_path = os.path.join(folder_path, f"{image_num}.pgm")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:

                img_vector = standardize_face(img)

                if image_num <= 8:
                    X_train.append(img_vector)
                    y_train.append(subject_id)
                else:
                    X_test.append(img_vector)
                    y_test.append(subject_id)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)