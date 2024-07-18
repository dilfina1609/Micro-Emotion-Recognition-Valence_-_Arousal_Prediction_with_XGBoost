import streamlit as st
import cv2
import dlib
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

# Load the PCA object and models
pca = joblib.load('pca_model.pkl')
model_arousal = joblib.load('model_arousal.pkl')
model_valence = joblib.load('model_valence.pkl')

# Function to extract landmarks from a frame
def extract_landmarks_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector(gray)
    
    # Assume only one face per frame and get the landmarks for the first face
    if len(faces) > 0:
        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        
        # Extract the coordinates of the landmarks
        landmark_coords = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmark_coords.append((x, y))
        
        return landmark_coords
    else:
        return None

# Function to preprocess a frame and extract features
def preprocess_frame(frame):
    landmarks = extract_landmarks_from_frame(frame)
    if landmarks is not None:
        landmarks_flat = np.array(landmarks).flatten()
        return landmarks_flat
    else:
        return None

# Function to predict arousal and valence from a frame
def predict_arousal_valence_from_frame(frame, model_arousal, model_valence, pca):
    features = preprocess_frame(frame)
    if features is not None:
        features_reduced = pca.transform([features])
        
        arousal_prediction = model_arousal.predict(features_reduced)
        valence_prediction = model_valence.predict(features_reduced)
        
        return arousal_prediction[0], valence_prediction[0]
    else:
        return None, None

# Create a directory to save the frames
output_dir = "predicted_frames"
os.makedirs(output_dir, exist_ok=True)

# Set up the webcam
video_capture = cv2.VideoCapture(0)

# Streamlit app layout
st.title("Arousal and Valence Prediction from Webcam")

# Placeholder to display video feed
video_placeholder = st.empty()

# Initialize predictions list
predictions = []

# Main loop for capturing frames and prediction
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to capture frame from webcam. Please check your webcam connection.")
        break

    # Predict arousal and valence
    arousal, valence = predict_arousal_valence_from_frame(frame, model_arousal, model_valence, pca)

    # Display prediction results on the frame
    if arousal is not None and valence is not None:
        cv2.putText(frame, f"Arousal: {arousal:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Valence: {valence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with OpenCV
    video_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Save the frame with predictions
    if arousal is not None and valence is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]  # Include milliseconds
        filename = f"frame_{timestamp}_arousal_{arousal:.2f}_valence_{valence:.2f}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        
        # Append prediction to the list
        predictions.append({'Timestamp': timestamp[:-3], 'Arousal': arousal, 'Valence': valence})

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()

# Convert list of dictionaries to DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to CSV
predictions_csv_path = os.path.join(output_dir, 'predictions.csv')
predictions_df.to_csv(predictions_csv_path, index=False)

# Display success message
st.success(f"Predictions saved to {predictions_csv_path}")

