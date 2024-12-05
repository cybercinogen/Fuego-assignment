import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Paths
input_csv_path = r"C:\Users\MOHITH\Desktop\fuego assignment\consolidated_influencer_data.csv"
output_plot_path = r"C:\Users\MOHITH\Desktop\fuego assignment\final_visualization_with_larger_faces.png"

def add_face_images(ax, influencer_data, image_size=0.25):
    """
    Adds representative face images at the end of each bar with a larger scale.
    Args:
        ax (Axes): Matplotlib Axes to add images to.
        influencer_data (DataFrame): Data containing face image paths.
        image_size (float): Scale of the images (default set to larger size).
    """
    for i, row in influencer_data.iterrows():
        face_path = row.get("Representative Face")
        if pd.notna(face_path) and os.path.exists(face_path):
            try:
                img = plt.imread(face_path)
                # Consistently larger scaling
                imagebox = OffsetImage(img, zoom=image_size)
                ab = AnnotationBbox(
                    imagebox,
                    (row["Average Performance"], i),  # Place image at the end of the bar
                    frameon=False,
                    box_alignment=(0.5, 0.5),  # Center alignment
                )
                ax.add_artist(ab)
            except Exception as e:
                print(f"Error loading image {face_path}: {e}")

def visualize_influencer_data(data_path, output_path, top_n=5):
    """
    Visualizes influencer performance with aligned larger representative faces.
    Args:
        data_path (str): Path to input CSV file with influencer data.
        output_path (str): Path to save the output visualization plot.
        top_n (int): Number of top performers to highlight.
    """
    # Ensure the input CSV file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found at: {data_path}")

    # Read the CSV file
    print(f"Reading data from {data_path}...")
    influencer_data = pd.read_csv(data_path)

    # Ensure required columns exist in the CSV
    required_columns = ["Unique Influencer", "Average Performance", "Representative Face"]
    for col in required_columns:
        if col not in influencer_data.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    influencer_data_sorted = influencer_data.sort_values("Average Performance", ascending=False)

    # Start plotting
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    bar_positions = range(len(influencer_data_sorted))

    # Highlight top performers in gold
    bars = ax.barh(
        bar_positions,
        influencer_data_sorted["Average Performance"],
        color=["gold" if i < top_n else "skyblue" for i in bar_positions],
        edgecolor="black",
    )

    # Set y-axis labels
    plt.yticks(bar_positions, influencer_data_sorted["Unique Influencer"])
    plt.xlabel("Average Performance")
    plt.ylabel("Unique Influencers")
    plt.title("Aligned Influencer Performance Visualization with Larger Faces")

    # Add face images at the end of each bar with larger scaling
    add_face_images(ax, influencer_data_sorted, image_size=0.25)

    # Add legend
    plt.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="gold", label="Top Performers"),
            plt.Rectangle((0, 0), 1, 1, color="skyblue", label="Others"),
        ],
        loc="lower right",
    )

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    try:
        visualize_influencer_data(input_csv_path, output_plot_path)
    except Exception as e:
        print(f"Error during visualization: {e}")
