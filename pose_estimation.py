import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find the pose landmarks.
    results = pose.process(image)

    # Convert the image back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the pose landmarks on the image.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image.
    cv2.imshow('Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
