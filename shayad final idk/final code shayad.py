import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp

# Dependencies:
# opencv-python
# numpy
# matplotlib
# mediapipe

def load_stereo_images(left_path, right_path):
    """
    Load the stereo image pair.
    """
    left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return left_img, right_img

def preprocess_images(left_img, right_img):
    """
    Preprocess the images to enhance features and reduce noise.
    """
    # Apply Gaussian blur to reduce noise
    left_img = cv2.GaussianBlur(left_img, (5, 5), 0)
    right_img = cv2.GaussianBlur(right_img, (5, 5), 0)
    
    # Apply histogram equalization to enhance contrast
    left_img = cv2.equalizeHist(left_img)
    right_img = cv2.equalizeHist(right_img)
    
    return left_img, right_img

def compute_disparity(left_img, right_img):
    """
    Compute the disparity map using Semi-Global Block Matching (SGBM).
    """
    # Create SGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # Compute disparity
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    
    return disparity

def reconstruct_3d(disparity, Q):
    """
    Reconstruct 3D points from disparity map.
    """
    # Reproject points to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Remove points with invalid depth
    mask = disparity > disparity.min()
    points_3d = points_3d[mask]
    
    return points_3d

def estimate_pose(image):
    """
    Estimate pose using MediaPipe.
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks
    else:
        return None

def visualize_results(left_img, disparity, points_3d, pose_landmarks):
    """
    Visualize the results: original image, disparity map, 3D reconstruction, and pose.
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Original left image
    ax1 = fig.add_subplot(221)
    ax1.imshow(left_img, cmap='gray')
    ax1.set_title('Left Image')
    ax1.axis('off')
    
    # Disparity map
    ax2 = fig.add_subplot(222)
    ax2.imshow(disparity, cmap='jet')
    ax2.set_title('Disparity Map')
    ax2.axis('off')
    
    # 3D reconstruction
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='jet', s=0.1)
    ax3.set_title('3D Reconstruction')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Pose estimation
    if pose_landmarks is not None:
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.scatter(pose_landmarks[:, 0], pose_landmarks[:, 1], pose_landmarks[:, 2])
        ax4.set_title('Pose Estimation')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def main(left_path, right_path):
    # Step 1: Load stereo images
    left_img, right_img = load_stereo_images(left_path, right_path)
    
    # Step 2: Preprocess images
    left_img_proc, right_img_proc = preprocess_images(left_img, right_img)
    
    # Step 3: Compute disparity
    disparity = compute_disparity(left_img_proc, right_img_proc)
    
    # Step 4: Reconstruct 3D scene
    # Note: Q matrix should be calibrated for your stereo setup
    # This is an example Q matrix, replace with your calibrated one
    Q = np.array([[1, 0, 0, -0.5*left_img.shape[1]],
                  [0, -1, 0, 0.5*left_img.shape[0]],
                  [0, 0, 0, -0.8*left_img.shape[1]],
                  [0, 0, 1/16, 0]])
    points_3d = reconstruct_3d(disparity, Q)
    
    # Step 5: Estimate pose
    pose_landmarks = estimate_pose(cv2.imread(left_path))
    
    # Step 6: Visualize results
    visualize_results(left_img, disparity, points_3d, pose_landmarks)

if __name__ == "__main__":
    left_image_path = "left_image.jpg"
    right_image_path = "right_image.jpg"
    main(left_image_path, right_image_path)