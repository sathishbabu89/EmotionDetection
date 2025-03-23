# Import libraries
import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

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

        # Create a placeholder for the video output
        video_placeholder = st.empty()

        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect emotions in the frame
            try:
                # Analyze the frame for emotions
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

                # Draw bounding box and dominant emotion on the frame
                for face in result:
                    x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Display the dominant emotion
                    dominant_emotion = max(face["emotion"], key=face["emotion"].get)
                    cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Convert the frame back to RGB for display in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                video_placeholder.image(frame_rgb, caption="Processed Frame", use_container_width=True)

            except Exception as e:
                st.error(f"Error detecting emotions: {e}")

        # Release the video capture object
        cap.release()

        # Delete the temporary video file
        os.unlink(video_path)

        st.write("Video processing complete.")