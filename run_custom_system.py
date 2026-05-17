# run_custom_system.py
import os
import cv2
import numpy as np
import feature_extraction
from preprocessing import standardize_face
from live_recognition import run_live_scenario

CUSTOM_DATASET_PATH = "custom_dataset"
LIVE_THRESHOLD = 4.6


def load_dynamic_dataset(dataset_path):
    """Dynamically loads faces and uses folder names as string labels."""
    X_train, y_train, X_test, y_test = [], [], [], []

    if not os.path.exists(dataset_path):
        return [], [], [], []

    # Get all subdirectories (which are now the names of the people)
    subject_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    for folder in subject_folders:
        # Convert folder name "John_Doe" back to label "John Doe"
        subject_name = folder.replace('_', ' ')
        folder_path = os.path.join(dataset_path, folder)
        images = [img for img in os.listdir(folder_path) if img.endswith('.pgm')]

        for i, image_name in enumerate(images):
            img_path = os.path.join(folder_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img_vector = standardize_face(img)
                if i < int(len(images) * 0.8):
                    X_train.append(img_vector)
                    y_train.append(subject_name)  # Passing the STRING name
                else:
                    X_test.append(img_vector)
                    y_test.append(subject_name)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def run_live_app():
    print(f"Scanning '{CUSTOM_DATASET_PATH}' for enrolled subjects...")
    X_train, y_train, X_test, y_test = load_dynamic_dataset(CUSTOM_DATASET_PATH)

    if len(X_train) == 0:
        print("Error: No data found. Run enrollment.py first!")
        return

    unique_subjects = len(np.unique(y_train))
    print(f"Found {unique_subjects} enrolled subjects: {np.unique(y_train)}")

    if unique_subjects < 2:
        print("CRITICAL: LDA requires at least 2 subjects to compare differences.")
        print("Please run enrollment.py again with a different name to enroll a second person.")
        return

    print("\nTraining HOG+LBP Fusion model on your custom face data...")
    # ---> CHANGED: Now using the Fusion Model instead of LDA <---
    fusion_train, fusion_test, fusion_pca = feature_extraction.train_fusion_model(X_train, y_train, X_test)

    print("\nStarting Live Security Camera...")
    run_live_scenario(
        trained_model=fusion_pca,
        train_features=fusion_train,
        train_labels=y_train,
        threshold=LIVE_THRESHOLD
    )


if __name__ == "__main__":
    run_live_app()