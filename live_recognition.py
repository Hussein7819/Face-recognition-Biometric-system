import cv2
import numpy as np
import dlib
from preprocessing import standardize_face, align_face
from feature_extraction import extract_hog_lbp_fusion


def run_live_scenario(trained_model, train_features, train_labels, threshold):
    print("Loading dlib facial landmark models...")
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    except RuntimeError:
        print("Error: 'shape_predictor_68_face_landmarks.dat' not found.")
        return

    cap = cv2.VideoCapture(0)
    print("Live Security Camera activated. Press 'q' to quit.")

    match_counter = 0
    REQUIRED_MATCHES = 5
    current_recognized_name = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            match_counter = 0
            current_recognized_name = None

        for rect in faces:
            shape = predictor(gray, rect)

            # --- REDUNDANCY REMOVED: Now using the unified alignment tool ---
            face_crop = align_face(frame, shape, rect)
            if face_crop.size == 0: continue

            face_flattened = standardize_face(face_crop).reshape(1, -1)
            face_fused = extract_hog_lbp_fusion(face_flattened)
            face_extracted = trained_model.transform(face_fused)

            distances = np.linalg.norm(train_features - face_extracted, axis=1)
            best_match_idx = np.argmin(distances)
            min_distance = distances[best_match_idx]
            person_name = train_labels[best_match_idx]

            if min_distance < threshold:
                if person_name == current_recognized_name:
                    match_counter += 1
                else:
                    current_recognized_name = person_name
                    match_counter = 1

                if match_counter >= REQUIRED_MATCHES:
                    text = f"{person_name} Granted (Dist: {min_distance:.1f})"
                    color = (0, 255, 0)
                else:
                    text = f"Verifying {person_name}... ({match_counter}/{REQUIRED_MATCHES})"
                    color = (0, 255, 255)
            else:
                match_counter = 0
                current_recognized_name = None
                text = f"UNKNOWN Denied (Dist: {min_distance:.1f})"
                color = (0, 0, 255)

            x1, y1 = max(0, rect.left()), max(0, rect.top())
            x2, y2 = min(frame.shape[1], rect.left() + rect.width()), min(frame.shape[0], rect.top() + rect.height())

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Live Biometrics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()