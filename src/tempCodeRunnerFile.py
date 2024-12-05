import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Paths
input_csv_path = r"C:\Users\MOHITH\Desktop\fuego assignment\consolidated_influencer_data.csv"
output_plot_path = r"C:\Users\MOHITH\Desktop\fuego assignment\optimized_visualization.png"

def add_scaled_face_images(ax, influencer_data, image_size=(50, 50)):
    """
    Adds scaled face images to the plot if paths are provided.
    """
    for i, row in influencer_data.iterrows():
        face_path = row.get("Representative Face")
        if pd.notna(face_path) and os.path.exists(face_path):
            try:
                img = Image.open(face_path).resize(image_size)  # Scale the image
                img = OffsetImage(img, zoom=1)
                ab = AnnotationBbox(
                    img,
                    (row["Average Performance"] + 0.1, i),  # Offset image to the right of the bar
                    frameon=False,
                    box_alignment=(0, 0.5),
                )
                ax.add_artist(ab)
            except Exception as e:
                print(f"Error loading image {face_path}: {e}")

def visualize_influencer_data(data_path, output_path, top_n=5):
    """
    Visualizes influencer performance with scaled representative faces.
    """
    influencer_data = pd.read_csv(data_path)
    influencer_data_sorted = influencer_data.sort_values("Average Performance", ascending=False)

    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    bar_positions = range(len(influencer_data_sorted))
    bars = ax.barh(
        bar_positions,
        influencer_data_sorted["Average Performance"],
        color=["gold" if i < top_n else "skyblue" for i in bar_positions],
        edgecolor="black",
    )

    plt.yticks(bar_positions, influencer_data_sorted["Influencer ID"])
    plt.xlabel("Average Performance")
    plt.ylabel("Unique Influencers")
    plt.title("Optimized Influencer Performance Visualization")

    add_scaled_face_images(ax, influencer_data_sorted)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_influencer_data(input_csv_path, output_plot_path)
