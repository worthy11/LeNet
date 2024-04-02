from functions import *

capture = cv2.VideoCapture(0)
img: cv2.Mat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands()

while(True):
    not_empty, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    if not_empty:
        results = hand.process(imgRGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for point in hand_landmarks.landmark:
                    print('Coords: {}, {}, {}'.format(point.x, point.y, point.z))
                print('\n')
        
        cv2.imshow('Image', img)
    key = cv2.waitKey(1)
