import numpy as np
from sklearn.cluster import DBSCAN
import os
import shutil

def cluster_faces_and_save_representative(face_embeddings, face_image_paths, output_dir):
    """
    Cluster face embeddings and save one representative face for each cluster.
    Args:
        face_embeddings (list): List of face embeddings.
        face_image_paths (list): Corresponding face image paths.
        output_dir (str): Directory to save representative faces.
    Returns:
        dict: Mapping of cluster IDs to representative face paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Validate embeddings
    if not face_embeddings:
        print("No face embeddings provided. Exiting clustering.")
        return {}

    face_embeddings = np.array(face_embeddings)
    if len(face_embeddings.shape) == 1:  # Handle single embedding case
        face_embeddings = face_embeddings.reshape(1, -1)

    clustering = DBSCAN(eps=0.5, min_samples=2).fit(face_embeddings)
    cluster_to_image = {}

    for cluster_id, image_path in zip(clustering.labels_, face_image_paths):
        if cluster_id == -1:
            continue
        if cluster_id not in cluster_to_image:
            cluster_to_image[cluster_id] = image_path
            shutil.copy(image_path, os.path.join(output_dir, f"{cluster_id}.png"))

    return cluster_to_image
