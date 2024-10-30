import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

# Video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Calculate distance between thumb tip and index fingertip
            thumb_tip = lmList[4]
            index_finger_tip = lmList[8]
            distance = calculate_distance(thumb_tip[1:], index_finger_tip[1:])

            # Map distance to brightness level (adjust as needed)
            brightness_level = int(distance * 100 / 200)  # Assuming maximum distance is 200

            # Adjust brightness (replace with appropriate command for your OS)
            # Example: For Windows, you might use pyautogui to simulate keystrokes
            pyautogui.hotkey('fn','F5', 'F6')  

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
