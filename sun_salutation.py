import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define Surya Namaskar sequence and landmarks to check
poses = [
    "Pranamasana (Prayer Pose)",
    "Hasta Uttanasana (Raised Arms Pose)",
    "Hasta Padasana (Hand to Foot Pose)",
    "Ashwa Sanchalanasana (Equestrian Pose)",
    "Dandasana (Stick Pose)",
    "Ashtanga Namaskara (Salute with Eight Parts)",
    "Bhujangasana (Cobra Pose)",
    "Adho Mukha Svanasana (Downward-Facing Dog Pose)",
]

current_step = 0
instructions_display_time = 5  # Seconds to display feedback
last_instruction_time = time.time()


# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Function to detect and instruct on specific pose
def detect_pose_landmarks(landmarks, step):
    if step == 0:  # Pranamasana (Prayer Pose)
        left_hand_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
        right_hand_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
        if abs(left_hand_x - right_hand_x) > 0.05:
            return "Bring your palms together at the center of your chest."
        return "Good! Hold the Prayer Pose."

    elif step == 1:  # Hasta Uttanasana (Raised Arms Pose)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        angle = calculate_angle(left_shoulder, right_shoulder, right_wrist)
        if angle < 160:
            return "Raise your arms straight up and arch slightly backward."
        return "Great! Arms are in position."

    elif step == 2:  # Hasta Padasana (Hand to Foot Pose)
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        if left_hand.y > left_foot.y and right_hand.y > right_foot.y:
            return "Bend forward more to touch your hands to your feet."
        return "Well done! Hands are near your feet."

    # Add similar logic for other poses...

    return "Hold this pose correctly!"


# Start Video Capture
cap = cv2.VideoCapture(0)

print("Starting Surya Namaskar guide...")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect poses
    results = pose.process(image)

    # Draw pose landmarks on the frame
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect the current pose and give instructions
        detected_pose = poses[current_step]
        feedback = detect_pose_landmarks(results.pose_landmarks.landmark, current_step)

        # Check time for pose switching
        if time.time() - last_instruction_time > instructions_display_time:
            current_step = (current_step + 1) % len(poses)
            last_instruction_time = time.time()

        # Display feedback and pose instructions
        cv2.putText(image, f"Pose: {detected_pose}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Feedback: {feedback}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow("Surya Namaskar Guide", image)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
