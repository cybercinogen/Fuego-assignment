import sys
import os
import pandas as pd

# Add the parent directory of 'src' to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_processing import extract_frames, detect_and_save_faces, save_representative_face
from src.hashing import compute_video_hash
from src.visualization import save_influencer_data_to_csv

# Paths
csv_path = r"C:\Users\MOHITH\Desktop\fuego assignment\data\raw_videos\Assignment Data - Sheet1.csv"
video_dir = r"C:\Users\MOHITH\Desktop\fuego assignment\data\videos"
frame_dir = r"C:\Users\MOHITH\Desktop\fuego assignment\data\frames"
face_dir = r"C:\Users\MOHITH\Desktop\fuego assignment\data\faces"
representative_face_dir = r"C:\Users\MOHITH\Desktop\fuego assignment\data\representative_faces"
output_csv_path = r"C:\Users\MOHITH\Desktop\fuego assignment\influencer_data_with_faces.csv"
missing_videos_path = r"C:\Users\MOHITH\Desktop\fuego assignment\missing_videos.csv"

# Step 1: Load metadata
metadata = pd.read_csv(csv_path)
metadata["Meta Tag"] = metadata["Video URL"].apply(lambda url: url.rstrip('/').split('/')[-1])
meta_to_performance = metadata.set_index("Meta Tag")["Performance"].to_dict()

# Step 2: Verify videos
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
video_data = []
missing_videos = []

for meta_tag, performance in meta_to_performance.items():
    video_file = f"{meta_tag}.mp4"
    video_path = os.path.join(video_dir, video_file)

    if os.path.exists(video_path):
        video_hash = compute_video_hash(video_path)
        video_data.append({"Meta Tag": meta_tag, "Video File": video_file, "Performance": performance, "Hash": video_hash})
    else:
        missing_videos.append({"Meta Tag": meta_tag, "Video URL": metadata.loc[metadata["Meta Tag"] == meta_tag, "Video URL"].values[0]})

# Save missing videos
if missing_videos:
    pd.DataFrame(missing_videos).to_csv(missing_videos_path, index=False)

# Step 3: Process all videos (no duplicates removed)
video_df = pd.DataFrame(video_data)
influencer_data = []

for _, row in video_df.iterrows():
    meta_tag = row["Meta Tag"]
    video_path = os.path.join(video_dir, row["Video File"])
    video_frame_dir = os.path.join(frame_dir, meta_tag)
    video_face_dir = os.path.join(face_dir, meta_tag)
    os.makedirs(video_frame_dir, exist_ok=True)
    os.makedirs(video_face_dir, exist_ok=True)

    # Recompute frames and faces
    extract_frames(video_path, video_frame_dir, fps=1)
    detected_faces = detect_and_save_faces(video_frame_dir, video_face_dir)

    if detected_faces:
        total_faces = len(detected_faces)
        representative_face_path = os.path.join(representative_face_dir, f"{meta_tag}.jpg")
        save_representative_face(video_face_dir, representative_face_path)
    else:
        total_faces = 0
        representative_face_path = None

    influencer_data.append({
        "Influencer ID": meta_tag,
        "Total Faces": total_faces,
        "Average Performance": row["Performance"],
        "Representative Face": representative_face_path
    })

# Step 4: Save data
pd.DataFrame(influencer_data).to_csv(output_csv_path, index=False)
print(f"Recomputed influencer data saved to {output_csv_path}")
