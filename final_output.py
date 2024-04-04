import pickle
import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Open the video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
dect = HandDetector(maxHands=1)
# Define labels dictionary
labels_dict = {'1': 'hello', '3': 'thankyou', '0':"Good Morning", "yes":"yes"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resize = cv2.resize(frame, (500, 500))
    H, W, _ = frame.shape
    hand, frame = dect.findHands(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract hand landmarks
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

            # Ensure data_aux has 84 features
            while len(data_aux) < 84:
                data_aux.append(0.0)  # Pad with zeros if needed

            # Predict using the model
            prediction = model.predict([np.asarray(data_aux)])
            print(prediction)

            # Get predicted character
            predicted_character = labels_dict[prediction[0]]
            print(predicted_character)

            # Draw rectangle and put text
            x1 = int(min(data_aux[::2]) * W) - 10
            y1 = int(min(data_aux[1::2]) * H) - 10
            x2 = int(max(data_aux[::2]) * W) - 10
            y2 = int(max(data_aux[1::2]) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
