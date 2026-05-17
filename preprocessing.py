import cv2
import numpy as np


def standardize_face(raw_face_image):
    """Standardizes raw cropped face images for the PCA/LDA biometric pipeline."""
    if len(raw_face_image.shape) == 3:
        gray = cv2.cvtColor(raw_face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw_face_image

    resized = cv2.resize(gray, (92, 112))
    normalized = cv2.equalizeHist(resized)
    return normalized.flatten()


def _get_eye_centroid(shape, indices):
    """Internal helper to calculate the center point of an eye."""
    pts = [shape.part(i) for i in indices]
    return (int(np.mean([p.x for p in pts])), int(np.mean([p.y for p in pts])))


def align_face(frame, shape, rect):
    """Performs an affine transformation to horizontally align eyes, then crops."""
    left_eye = _get_eye_centroid(shape, range(36, 42))
    right_eye = _get_eye_centroid(shape, range(42, 48))

    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)

    x1, y1 = max(0, rect.left()), max(0, rect.top())
    x2, y2 = min(frame.shape[1], rect.left() + rect.width()), min(frame.shape[0], rect.top() + rect.height())

    return aligned_frame[y1:y2, x1:x2]