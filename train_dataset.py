import os
import cv2
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

path = 'dataset'

data = []
labels = []




for dir in os.listdir(path):  # this part of for loop was working in list folder img
    print(dir)                # print ths img name in list
    for img in os.listdir(os.path.join(path, dir)):  # this part of for loop list the folder name and inside a folder imgs using [:1] spilt the img
        print(img)                                 # print the folder imgname
        data_axi = []
        img1 = cv2.imread(os.path.join(path, dir, img))
        # print(img1)

        img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     img_rgb,  # image to draw
                #     hand_landmarks,  # model output
                #     mp_hands.HAND_CONNECTIONS,  # hand connections
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())
                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y =hand_landmarks.landmark[i].y
                    data_axi.append(x)
                    data_axi.append(y)
                data.append(data_axi)
                labels.append(dir)

                # print(data)
                # print(labels)

#
# #         plt.figure()
# #         plt.imshow(img_rgb)
# #
# # plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()