# Import libraries
import streamlit as st
from deepface import DeepFace
from PIL import Image
import cv2
import numpy as np

# Set up the Streamlit app title
st.title("Emotion Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to OpenCV format
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Detect emotions using DeepFace
    try:
        # Analyze the image for emotions
        result = DeepFace.analyze(image_cv2, actions=["emotion"], enforce_detection=False)

        # Display the detected emotions
        st.write("Detected Emotions:")
        st.write(result[0]["emotion"])

        # Draw bounding box and dominant emotion on the image
        for face in result:
            x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
            cv2.rectangle(image_cv2, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display the dominant emotion
            dominant_emotion = max(face["emotion"], key=face["emotion"].get)
            cv2.putText(image_cv2, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert the image back to RGB for display in Streamlit
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Display the resulting image
        st.image(image_cv2, caption="Emotion Detection Result", use_column_width=True)

    except Exception as e:
        st.error(f"Error detecting emotions: {e}")