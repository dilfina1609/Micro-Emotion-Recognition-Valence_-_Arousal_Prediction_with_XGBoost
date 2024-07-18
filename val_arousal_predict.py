import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import requests
import bz2
import cv2
import dlib

# Load the CSV file
file_path = r'C:\Users\Dilfina\OneDrive\Desktop\av\output.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())
print(data.columns)

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\Dilfina\OneDrive\Desktop\av\output.csv'
data = pd.read_csv(file_path)

# Calculate upper and lower bounds for arousal and valence
arousal_min = data['arousal'].min()
arousal_max = data['arousal'].max()
valence_min = data['valence'].min()
valence_max = data['valence'].max()

print(f"Arousal: Min={arousal_min}, Max={arousal_max}")
print(f"Valence: Min={valence_min}, Max={valence_max}")

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data['arousal'], bins=20, color='blue', alpha=0.7)
plt.axvline(arousal_min, color='red', linestyle='dashed', linewidth=1, label='Min')
plt.axvline(arousal_max, color='green', linestyle='dashed', linewidth=1, label='Max')
plt.xlabel('Arousal')
plt.ylabel('Frequency')
plt.title('Distribution of Arousal')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(data['valence'], bins=20, color='orange', alpha=0.7)
plt.axvline(valence_min, color='red', linestyle='dashed', linewidth=1, label='Min')
plt.axvline(valence_max, color='green', linestyle='dashed', linewidth=1, label='Max')
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.title('Distribution of Valence')
plt.legend()

plt.tight_layout()
plt.show()

# Extract features from landmarks
def extract_landmark_features(landmarks):
    return np.array([np.array(eval(landmark)).flatten() for landmark in landmarks])

# Extract features and target variables
X = extract_landmark_features(data['landmarks'])
y_arousal = data['arousal']
y_valence = data['valence']

# Split the data into training and testing sets
X_train_arousal, X_test_arousal, y_train_arousal, y_test_arousal = train_test_split(X, y_arousal, test_size=0.2, random_state=42)
X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(X, y_valence, test_size=0.2, random_state=42)
# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)  # Reducing to 50 components
X_train_reduced = pca.fit_transform(X_train_arousal)
X_test_reduced = pca.transform(X_test_arousal)

# Save the PCA object
joblib.dump(pca, 'pca_model.pkl')
# Initialize and train the XGBoost models
model_arousal = xgb.XGBRegressor(max_depth=3, n_estimators=100, verbosity=1)
model_valence = xgb.XGBRegressor(max_depth=3, n_estimators=100, verbosity=1)

model_arousal.fit(X_train_reduced, y_train_arousal)
model_valence.fit(X_train_reduced, y_train_valence)
# Make predictions and evaluate the models
y_pred_arousal = model_arousal.predict(X_test_reduced)
y_pred_valence = model_valence.predict(X_test_reduced)

mse_arousal = mean_squared_error(y_test_arousal, y_pred_arousal)
mse_valence = mean_squared_error(y_test_valence, y_pred_valence)

print(f"MSE for arousal: {mse_arousal}")
print(f"MSE for valence: {mse_valence}")
# Grid search for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search_arousal = GridSearchCV(estimator=model_arousal, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search_valence = GridSearchCV(estimator=model_valence, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)

grid_search_arousal.fit(X_train_reduced, y_train_arousal)
grid_search_valence.fit(X_train_reduced, y_train_valence)

best_params_arousal = grid_search_arousal.best_params_
best_params_valence = grid_search_valence.best_params_

print(f"Best parameters for arousal: {best_params_arousal}")
print(f"Best parameters for valence: {best_params_valence}")
# Evaluate the models
r2_arousal = r2_score(y_test_arousal, y_pred_arousal)
r2_valence = r2_score(y_test_valence, y_pred_valence)

print(f"R-squared for arousal: {r2_arousal}")
print(f"R-squared for valence: {r2_valence}")
# Plotting predictions vs actual values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_arousal, y_pred_arousal)
plt.xlabel("Actual Arousal")
plt.ylabel("Predicted Arousal")
plt.title("Arousal: Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test_valence, y_pred_valence)
plt.xlabel("Actual Valence")
plt.ylabel("Predicted Valence")
plt.title("Valence: Actual vs Predicted")

plt.show()
# Save the models
joblib.dump(model_arousal, 'model_arousal.pkl')
joblib.dump(model_valence, 'model_valence.pkl')
# Download and decompress the shape predictor file
url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
response = requests.get(url)
with open('shape_predictor_68_face_landmarks.dat.bz2', 'wb') as file:
    file.write(response.content)

# Decompress the file
with open('shape_predictor_68_face_landmarks.dat.bz2', 'rb') as file:
    decompressed_data = bz2.decompress(file.read())
with open('shape_predictor_68_face_landmarks.dat', 'wb') as file:
    file.write(decompressed_data)

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)
def extract_landmarks_from_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_detector(gray)
    
    # Assume only one face per image and get the landmarks for the first face
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
        raise ValueError("No faces detected in the image")
# Redefine the preprocessing and prediction functions
def preprocess_new_image(image_path):
    landmarks = extract_landmarks_from_image(image_path)
    landmarks_flat = np.array(landmarks).flatten()
    return landmarks_flat

def predict_arousal_valence(image_path, model_arousal, model_valence, pca):
    features = preprocess_new_image(image_path)
    features_reduced = pca.transform([features])
    
    arousal_prediction = model_arousal.predict(features_reduced)
    valence_prediction = model_valence.predict(features_reduced)
    
    return arousal_prediction[0], valence_prediction[0]
# Load the PCA object and models
pca = joblib.load('pca_model.pkl')
model_arousal = joblib.load('model_arousal.pkl')
model_valence = joblib.load('model_valence.pkl')

# Example usage
image_path = r'C:\Users\Dilfina\OneDrive\Desktop\av\test.jpeg'
arousal, valence = predict_arousal_valence(image_path, model_arousal, model_valence, pca)
print(f"Arousal: {arousal}, Valence: {valence}")
import cv2
import dlib
import numpy as np
import joblib
import os
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

def preprocess_frame(frame):
    landmarks = extract_landmarks_from_frame(frame)
    if landmarks is not None:
        landmarks_flat = np.array(landmarks).flatten()
        return landmarks_flat
    else:
        return None

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

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    arousal, valence = predict_arousal_valence_from_frame(frame, model_arousal, model_valence, pca)
    
    if arousal is not None and valence is not None:
        cv2.putText(frame, f"Arousal: {arousal:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Valence: {valence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Save the frame with predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        frame_filename = os.path.join(output_dir, f"frame_{timestamp}_arousal_{arousal:.2f}_valence_{valence:.2f}.jpg")
        cv2.imwrite(frame_filename, frame)
    
    # Display the frame with predictions
    cv2.imshow('Arousal and Valence Prediction', frame)
    
    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
