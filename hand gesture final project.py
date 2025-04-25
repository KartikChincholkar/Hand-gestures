import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Load trained model
model = load_model(r"D:\1LPUcollegesem\semister 6\INT 345 Computer vision\asl_model1.h5")

# Class labels: A-Z + Space + Nothing
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Space", "Nothing"]

# Text-to-speech engine
engine = pyttsx3.init()

# MediaPipe hand detection setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Webcam setup
cap = cv2.VideoCapture(0)
prev_label = ""
sentence = ""
last_spoken = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw landmarks and detect hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Region of Interest (ROI)
        x1, y1, x2, y2 = 100, 100, 324, 324
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (64, 64))
        roi_normalized = roi_resized / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)

        # Predict gesture
        prediction = model.predict(roi_input)
        confidence = np.max(prediction)
        if confidence > 0.80:  # only accept if confident
            label = classes[np.argmax(prediction)]
        else:
            label = "Uncertain"

        # Display current gesture
        cv2.putText(frame, f"Gesture: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak if gesture changes and not Nothing
        if label != prev_label and label != "Nothing":
            if label == "Space":
                sentence += " "
            else:
                sentence += label
            prev_label = label
            engine.say(label)
            engine.runAndWait()
            last_spoken = time.time()

    # Draw the ROI box
    cv2.rectangle(frame, (100, 100), (324, 324), (255, 0, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Gesture to Speech", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
