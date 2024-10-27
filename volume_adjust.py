import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to adjust brightness based on gesture
def adjust_brightness(index_finger_tip_y, middle_finger_tip_y):
    if index_finger_tip_y < middle_finger_tip_y:
        pyautogui.press('volumeup')
    elif index_finger_tip_y > middle_finger_tip_y:
        pyautogui.press('volumedown')

# Video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Hand landmark detection
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Extract index finger tip and middle finger tip
            index_finger_tip_y = lmList[8][2]
            middle_finger_tip_y = lmList[12][2]

            # Adjust brightness based on gesture
            adjust_brightness(index_finger_tip_y, middle_finger_tip_y)

            # Draw landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
