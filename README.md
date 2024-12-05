# Fuego Assignment

## Overview

This project is focused on analyzing influencer performance from a collection of video data. The goal is to process video files, extract frames, detect faces, calculate influencer performance, and visualize the results. The project makes use of Python libraries like `pandas`, `opencv`, and `face_recognition` for various tasks including video processing, face detection, and data manipulation.

## Assumptions

- Each influencer has multiple videos, but only the first detected face is considered the "representative" face.
- The dataset assumes the videos are properly formatted and available for analysis.
- The assignment was designed to handle a batch of 100 videos at a time, and scalability considerations for handling larger data are only addressed theoretically.
- Performance scores and face detection are assumed to be accurate based on available data, and further validation of the algorithm's accuracy could be added in a production system.

## Technologies Used

- **Python:** Main programming language
- **Pandas:** For data manipulation and CSV handling
- **OpenCV:** For video frame extraction
- **Face Recognition:** For detecting and processing faces
- **Matplotlib:** For creating visualizations
- **Scikit-learn (optional):** For clustering and machine learning models (used conceptually)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/cybercinogen/Fuego-assignment.git
   cd Fuego-assignment
## Result
- The final result can be seen here "final_visualization_with_larger_faces.png"
