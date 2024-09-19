import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def capture_pose():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return results.pose_landmarks

def reconstruct_3d_pose(landmarks):
    x = []
    y = []
    z = []
    for landmark in landmarks.landmark:
        x.append(landmark.x)
        y.append(landmark.y)
        z.append(landmark.z)
    
    return np.array(x), np.array(y), np.array(z)

def visualize_3d_pose(x, y, z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    
    # Connect joints to form a stick figure
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        ax.plot([x[start_idx], x[end_idx]], 
                [y[start_idx], y[end_idx]], 
                [z[start_idx], z[end_idx]])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Reconstruction')
    plt.show()

def main():
    print("Capturing pose from webcam. Press 'q' to capture and reconstruct.")
    landmarks = capture_pose()
    if landmarks:
        x, y, z = reconstruct_3d_pose(landmarks)
        visualize_3d_pose(x, y, z)
    else:
        print("No pose detected. Please try again.")

if __name__ == "__main__":
    main()