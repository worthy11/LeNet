import cv2
import numpy as np
import mediapipe as mp

def Preprocess(img: cv2.Mat) -> cv2.Mat:
    imgPre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgPre = cv2.GaussianBlur(imgPre, (9, 9), 5)
    imgPre = cv2.Canny(imgPre, 25., 75.)

    cv2.imshow('Mask', imgPre)
    contours, hierarchy = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = list()
    cv2.drawContours(img, contours, -1, (255, 0, 255), 2, 1)
    for contour in contours:
        # area = cv2.contourArea(contour, False)
        # peri = cv2.arcLength(contour, True)
        # poly = cv2.approxPolyDP(contours, .02*peri, True)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img

def main():
    capture = cv2.VideoCapture(0)
    img: cv2.Mat
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand = mp_hands.Hands()
    
    while(True):
        not_empty, img = capture.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not_empty:
            hands = hand.process(imgRGB)
            if hands.multi_hand_landmarks:
                for h in hands.multi_hand_landmarks:
                    coords = np.array(h)
                    mp_drawing.draw_landmarks(img, h, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Image', img)
        cv2.waitKey(1)