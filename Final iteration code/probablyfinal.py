import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Function to extract landmarks from an image
def extract_landmarks(pose_landmarks):
    landmarks = []
    for landmark in pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks)

# Load dataset and extract landmarks
def load_dataset_with_landmarks(folder_path):
    X = []
    y = []
    label_map = {'standing': 0, 'sitting': 1}  # Labels for standing and sitting

    for label in label_map:
        label_folder = os.path.join(folder_path, label)
        if not os.path.exists(label_folder):
            print(f"Warning: Folder {label_folder} does not exist.")
            continue

        for filename in os.listdir(label_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_folder, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                with mp_pose.Pose(static_image_mode=True) as pose:
                    results = pose.process(image_rgb)
                    if results.pose_landmarks:
                        landmarks = extract_landmarks(results.pose_landmarks)
                        X.append(landmarks)
                        y.append(label_map[label])

    return np.array(X), np.array(y)

# Train RandomForest model using pose landmarks as features
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    return clf

# Function to visualize 3D pose landmarks using estimated depth
def visualize_3d_pose(pose_landmarks):
    x = []
    y = []
    z = []
    # Estimation of depth based on some predefined proportions
    depth_estimate = {
        "nose": 0.5,
        "left_shoulder": 0.6,
        "right_shoulder": 0.6,
        "left_hip": 0.8,
        "right_hip": 0.8,
        "left_knee": 1.0,
        "right_knee": 1.0,
    }

    for idx, landmark in enumerate(pose_landmarks.landmark):
        # Map 2D coordinates to a 3D space
        x.append(landmark.x)  # x coordinate
        y.append(landmark.y)  # y coordinate
        # Assign depth based on the landmark
        z.append(depth_estimate.get(mp_pose.PoseLandmark(idx).name, 0.5))  # Default depth if not specified

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o')

    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], [z[start_idx], z[end_idx]], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Reconstruction from a Single Image')
    plt.show()

# Function to predict pose of a new image
def predict_pose(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = extract_landmarks(results.pose_landmarks)
            prediction = model.predict([landmarks])  # Predict using the trained model

            if prediction[0] == 0:
                print("Predicted Pose: Standing")
            else:
                print("Predicted Pose: Sitting")

            # Visualize the 3D pose
            visualize_3d_pose(results.pose_landmarks)
        else:
            print("No landmarks detected.")

# Main function
def main():
    dataset_path = "C:/Users/ACER/Desktop/Pose Estimation/bhai final code/ProData/"  # Adjust this path to your dataset
    X, y = load_dataset_with_landmarks(dataset_path)

    if len(X) == 0:
        print("No images were loaded. Please check the dataset folder structure and image formats.")
        return

    print(f"Loaded {len(X)} images for training.")
    model = train_model(X, y)

    # Test the model on a new image
    test_image_path = "C:/Users/ACER/Desktop/Pose Estimation/bhai final code/ProData/Standing/stand 1.jpg"  # Adjust this path to your test image
    predict_pose(test_image_path, model)

if __name__ == "__main__":
    main()