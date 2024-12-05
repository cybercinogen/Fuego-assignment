import os
import cv2
import face_recognition
from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np

def extract_frames(video_path, frame_output_dir, fps=1):
    """
    Extracts frames from a video and saves them to the specified directory.
    """
    os.makedirs(frame_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(frame_rate / fps))
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frame_output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {frame_output_dir}")


def detect_and_save_faces(frame_dir, face_output_dir):
    """
    Detects faces in frames and saves them to the specified directory.
    Returns a list of detected face file paths.
    """
    os.makedirs(face_output_dir, exist_ok=True)
    detected_faces = []
    for frame_file in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error reading frame: {frame_path}")
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face = frame[top:bottom, left:right]
            face_path = os.path.join(face_output_dir, f"{frame_file.split('.')[0]}_face_{i}.jpg")
            cv2.imwrite(face_path, face)
            detected_faces.append(face_path)

    print(f"Saved {len(detected_faces)} faces from frames in {frame_dir} to {face_output_dir}")
    return detected_faces


def save_representative_face(face_output_dir, representative_face_path):
    """
    Saves the first detected face as the representative face for an influencer.
    """
    face_files = [os.path.join(face_output_dir, f) for f in os.listdir(face_output_dir) if f.endswith(".jpg")]
    if not face_files:
        print(f"No faces found in {face_output_dir}. Unable to save representative face.")
        return

    representative_face = cv2.imread(face_files[0])
    os.makedirs(os.path.dirname(representative_face_path), exist_ok=True)
    cv2.imwrite(representative_face_path, representative_face)
    print(f"Representative face saved to {representative_face_path}")


def group_faces_by_identity(face_paths, eps=0.5, min_samples=1):
    """
    Groups faces based on identity using DBSCAN clustering on face encodings.

    Args:
        face_paths (list): List of file paths to face images.
        eps (float): DBSCAN clustering epsilon parameter (distance threshold).
        min_samples (int): DBSCAN clustering minimum samples parameter.

    Returns:
        dict: A mapping of group IDs to lists of face paths.
    """
    if not face_paths:
        print("No face paths provided for grouping.")
        return {}

    print(f"Processing {len(face_paths)} faces for grouping...")
    encodings = []
    valid_face_paths = []

    for face_path in face_paths:
        try:
            image = face_recognition.load_image_file(face_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                valid_face_paths.append(face_path)
        except Exception as e:
            print(f"Error processing face {face_path}: {e}")

    if not encodings:
        print("No valid face encodings found.")
        return {}

    encodings = np.array(encodings)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(encodings)
    labels = clustering.labels_

    # Map group IDs to face paths
    grouped_faces = defaultdict(list)
    for face_path, label in zip(valid_face_paths, labels):
        grouped_faces[label].append(face_path)

    print(f"Grouped faces into {len(grouped_faces)} unique identities.")
    return dict(grouped_faces)
