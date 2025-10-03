import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mask_detector.model")

model = load_model()

st.title("😷 Face Mask Detector")
st.write("Detect if a person is wearing a mask in real-time using your webcam.")

# Webcam option
run = st.checkbox("Run Webcam")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Webcam not accessible.")
        break

    # Preprocess frame
    face = cv2.resize(frame, (224, 224))  # adjust size as per training
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    # Prediction
    pred = model.predict(face)[0]
    label = "Mask" if pred[0] > 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

    # Display result on frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

cap.release()
