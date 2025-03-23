# Import libraries
import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd

# Set up the Streamlit app title
st.title("Emotion Detection App")

# Upload video through Streamlit
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        st.error("Error opening video file.")
    else:
        st.write("Video uploaded successfully. Processing frames...")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Optimize performance: Reduce frame rate and resize frames
        target_frame_rate = 10  # Process 10 frames per second
        frame_skip = int(frame_rate / target_frame_rate)
        resize_scale = 0.5  # Resize frames to 50% of original size

        # Initialize variables for emotion statistics
        emotion_stats = {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "neutral": 0,
        }
        total_faces = 0

        # Create a temporary file for the processed video
        output_path = "processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, target_frame_rate, (int(frame_width * resize_scale), int(frame_height * resize_scale)))

        # Create a placeholder for the video output
        video_placeholder = st.empty()

        # Process each frame of the video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce processing load
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Resize the frame for faster processing
            frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

            # Detect emotions in the frame
            try:
                # Analyze the frame for emotions
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

                # Update emotion statistics
                for face in result:
                    total_faces += 1
                    for emotion, score in face["emotion"].items():
                        emotion_stats[emotion] += score

                # Draw bounding box and dominant emotion on the frame
                for face in result:
                    x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Display the dominant emotion
                    dominant_emotion = max(face["emotion"], key=face["emotion"].get)
                    cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Write the processed frame to the output video
                out.write(frame)

                # Convert the frame back to RGB for display in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                video_placeholder.image(frame_rgb, caption="Processed Frame", use_container_width=True)

            except Exception as e:
                st.error(f"Error detecting emotions: {e}")

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Delete the temporary video file
        os.unlink(video_path)

        st.write("Video processing complete.")

        # Calculate average emotion scores
        if total_faces > 0:
            for emotion in emotion_stats:
                emotion_stats[emotion] /= total_faces

            # Display emotion statistics
            st.write("### Emotion Statistics")
            st.write(pd.DataFrame.from_dict(emotion_stats, orient="index", columns=["Average Score"]))

        # Provide a download link for the processed video
        st.write("### Download Processed Video")
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

        # Clean up the processed video file
        os.unlink(output_path)