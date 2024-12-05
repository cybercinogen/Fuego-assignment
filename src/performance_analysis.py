import os
import sys
import pandas as pd
from video_processing import group_faces_by_identity  # Ensure this function exists in video_processing.py

# Paths
input_csv_path = r"C:\Users\MOHITH\Desktop\fuego assignment\influencer_data_with_faces.csv"
output_csv_path = r"C:\Users\MOHITH\Desktop\fuego assignment\consolidated_influencer_data.csv"

def consolidate_by_identity(input_path, output_path):
    """
    Consolidates influencer data by grouping influencers based on face similarity.
    Args:
        input_path (str): Path to the input CSV file with influencer data.
        output_path (str): Path to save the consolidated influencer data.
    """
    print(f"Reading data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: Input CSV file not found at {input_path}")
        return

    # Read input CSV
    data = pd.read_csv(input_path)
    print("Data read successfully. First few rows:")
    print(data.head())

    # Check if "Representative Face" column exists
    if "Representative Face" not in data.columns:
        print(f"Error: 'Representative Face' column not found in input CSV.")
        return

    # Drop rows with missing representative face paths
    face_paths = data["Representative Face"].dropna()
    if face_paths.empty:
        print("Error: No valid representative face paths found in the input data.")
        return

    print("Grouping faces by identity...")
    grouped_faces = group_faces_by_identity(face_paths.tolist())  # Group by face similarity
    print(f"Grouped {len(grouped_faces)} unique identities.")

    # Map influencers to identity groups
    identity_map = {face: group_id for group_id, faces in grouped_faces.items() for face in faces}
    data["Identity Group"] = data["Representative Face"].map(identity_map)

    if data["Identity Group"].isnull().any():
        print("Warning: Some rows could not be mapped to an identity group.")

    # Aggregate metrics for each unique identity
    print("Aggregating data by identity group...")
    consolidated = data.groupby("Identity Group").agg({
        "Total Faces": "sum",
        "Average Performance": "mean",
        "Representative Face": "first",  # Choose one face as the representative
    }).reset_index()

    consolidated.rename(columns={"Identity Group": "Unique Influencer"}, inplace=True)
    print("Consolidated data preview:")
    print(consolidated.head())

    # Save consolidated data to CSV
    print(f"Saving consolidated data to {output_path}...")
    consolidated.to_csv(output_path, index=False)
    print(f"Consolidated influencer data saved to {output_path}")

if __name__ == "__main__":
    consolidate_by_identity(input_csv_path, output_csv_path)
