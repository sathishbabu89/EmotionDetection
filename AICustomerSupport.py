# Import libraries
import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

# Recommendation function
def get_recommendations(emotion_stats):
    dominant_emotion = max(emotion_stats, key=emotion_stats.get)
    recommendation = ""
    action_items = []
    
    if dominant_emotion in ['happy', 'neutral']:
        recommendation = "Customer is in positive state"
        action_items = [
            "Offer promotional products",
            "Suggest premium upgrades",
            "Request feedback or review",
            "Recommend complementary items"
        ]
    elif dominant_emotion in ['angry', 'sad']:
        recommendation = "Customer might need support"
        action_items = [
            "Escalate to senior staff",
            "Offer discount or apology",
            "Provide return/refund options",
            "Connect with customer support"
        ]
    elif dominant_emotion in ['fear', 'surprise']:
        recommendation = "Customer needs reassurance"
        action_items = [
            "Provide detailed product information",
            "Offer live assistance",
            "Share testimonials/reviews",
            "Highlight security/safety features"
        ]
    elif dominant_emotion == 'disgust':
        recommendation = "Customer seems dissatisfied"
        action_items = [
            "Offer alternative products",
            "Explain quality control processes",
            "Provide customization options",
            "Request specific feedback"
        ]
    
    return {
        "dominant_emotion": dominant_emotion,
        "recommendation": recommendation,
        "action_items": action_items
    }

# Set up the Streamlit app title
st.title("Emotion Detection & Recommendation Engine")

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

        # Optimize performance
        target_frame_rate = 10
        frame_skip = int(frame_rate / target_frame_rate)
        resize_scale = 0.5

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
        emotion_history = []
        frame_timestamps = []

        # Create output video
        output_path = "processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, target_frame_rate, 
                            (int(frame_width * resize_scale), int(frame_height * resize_scale)))

        # Create placeholders
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        chart_placeholder = st.empty()
        recommendation_placeholder = st.empty()

        # Process each frame
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                current_emotions = {e: 0 for e in emotion_stats}
                current_faces = 0
                
                for face in result:
                    current_faces += 1
                    for emotion, score in face["emotion"].items():
                        emotion_stats[emotion] += score
                        current_emotions[emotion] += score

                total_faces += current_faces
                
                if current_faces > 0:
                    for emotion in current_emotions:
                        current_emotions[emotion] /= current_faces
                    emotion_history.append(current_emotions)
                    frame_timestamps.append(frame_count / frame_rate)

                # Draw bounding boxes
                for face in result:
                    x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    dominant_emotion = max(face["emotion"], key=face["emotion"].get)
                    cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                out.write(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, caption="Processed Frame", use_container_width=True)
                
                # Update displays
                if total_faces > 0:
                    current_stats = {e: emotion_stats[e]/total_faces for e in emotion_stats}
                    stats_placeholder.dataframe(pd.DataFrame.from_dict(current_stats, orient="index", columns=["Average Score"]))
                    
                    # Get recommendations
                    recommendations = get_recommendations(current_stats)
                    
                    # Display recommendations
                    with recommendation_placeholder.container():
                        st.subheader("Recommendations Based on Emotions")
                        st.write(f"**Dominant Emotion:** {recommendations['dominant_emotion'].capitalize()}")
                        st.write(f"**Analysis:** {recommendations['recommendation']}")
                        st.write("**Suggested Actions:**")
                        for item in recommendations['action_items']:
                            st.write(f"- {item}")
                
                # Update chart
                if len(emotion_history) > 0 and len(emotion_history) % 5 == 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    for emotion in emotion_stats:
                        emotion_values = [frame[emotion] for frame in emotion_history]
                        ax.plot(frame_timestamps[:len(emotion_values)], emotion_values, label=emotion)
                    ax.set_title("Emotion Trends Over Time")
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Emotion Score")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True)
                    chart_placeholder.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Error detecting emotions: {e}")

        # Clean up
        cap.release()
        out.release()
        os.unlink(video_path)

        processing_time = time.time() - start_time
        st.write(f"Video processing complete. Time taken: {processing_time:.2f} seconds")

        # Final results
        if total_faces > 0:
            for emotion in emotion_stats:
                emotion_stats[emotion] /= total_faces

            st.write("### Final Emotion Statistics")
            st.dataframe(pd.DataFrame.from_dict(emotion_stats, orient="index", columns=["Average Score"]))
            
            # Final recommendations
            final_recommendations = get_recommendations(emotion_stats)
            st.subheader("Final Recommendations")
            st.write(f"**Dominant Emotion:** {final_recommendations['dominant_emotion'].capitalize()}")
            st.write(f"**Analysis:** {final_recommendations['recommendation']}")
            st.write("**Suggested Actions:**")
            for item in final_recommendations['action_items']:
                st.write(f"- {item}")
            
            # Final chart
            st.write("### Final Emotion Trends")
            fig, ax = plt.subplots(figsize=(10, 4))
            for emotion in emotion_stats:
                emotion_values = [frame[emotion] for frame in emotion_history]
                ax.plot(frame_timestamps[:len(emotion_values)], emotion_values, label=emotion)
            ax.set_title("Final Emotion Trends Over Time")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Emotion Score")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
            st.pyplot(fig)

        # Download processed video
        st.write("### Download Processed Video")
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

        os.unlink(output_path)
