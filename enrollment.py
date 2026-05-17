import cv2
import dlib
import os
from preprocessing import align_face


def enroll_new_subject(subject_name, num_images=10):
    """Captures and aligns faces from the webcam to create a custom dataset."""
    safe_name = subject_name.replace(" ", "_")
    save_dir = f"custom_dataset/{safe_name}"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dlib models...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    print(f"\nEnrollment started for: {subject_name}")
    print(f"Look at the camera. Capturing {num_images} images. Press 'c' to capture a frame.")

    count = 1
    while count <= num_images:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Capturing {subject_name}: {count}/{num_images} - Press 'C'",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for rect in faces:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(display_frame, (max(0, x), max(0, y)),
                          (min(frame.shape[1], x + w), min(frame.shape[0], y + h)), (255, 0, 0), 2)

        cv2.imshow("Enrollment", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and len(faces) > 0:
            rect = faces[0]
            shape = predictor(gray, rect)

            # --- REDUNDANCY REMOVED: Now using the unified alignment tool ---
            face_crop = align_face(frame, shape, rect)

            if face_crop.size > 0:
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                file_path = os.path.join(save_dir, f"{count}.pgm")
                cv2.imwrite(file_path, face_gray)
                print(f"Saved {file_path}")
                count += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Enrollment complete for {subject_name}.")


if __name__ == "__main__":
    print("\n--- Biometric Face Enrollment ---")
    user_input = input("Enter the full name of the person: ").strip()
    if user_input:
        enroll_new_subject(subject_name=user_input, num_images=10)
    else:
        print("Error: Name cannot be blank.")