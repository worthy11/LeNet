import cv2
import numpy as np
import mediapipe as mp
import csv
import os

letters = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    # 16: 'Q',
    16: 'R',
    17: 'S',
    18: 'T',
    19: 'U',
    20: 'V',
    21: 'W',
    22: 'X',
    23: 'Y',
    24: 'Z'
}
landmarks = {
    0: 'WRIST',
    1: 'THUMB_CMC',
    2: 'THUMB_MCP',
    3: 'THUMB_IP',
    4: 'THUMB_TIP',
    5: 'INDEX_FINGER_MCP',
    6: 'INDEX_FINGER_PIP',
    7: 'INDEX_FINGER_DIP',
    8: 'INDEX_FINGER_TIP',
    9: 'MIDDLE_FINGER_MCP',
    10: 'MIDDLE_FINGER_PIP',
    11: 'MIDDLE_FINGER_DIP',
    12: 'MIDDLE_FINGER_TIP',
    13: 'RING_FINGER_MCP',
    14: 'RING_FINGER_PIP',
    15: 'RING_FINGER_DIP',
    16: 'RING_FINGER_TIP',
    17: 'PINKY_MCP',
    18: 'PINKY_PIP',
    19: 'PINKY_DIP',
    20: 'PINKY_TIP'
}
threshold = 0.01

def ComputeDistances(arr: np.array, ndims: int) -> np.array:
    if ndims == 2:
        w = np.max(arr[:, 0]) - np.min(arr[:, 0])
        h = np.max(arr[:, 1]) - np.min(arr[:, 1])
        distances = np.empty((21, 21))
        for i in range(21):
            for j in range(21):
                dx = (arr[j][0] - arr[i][0]) / w
                dy = (arr[j][1] - arr[i][1]) / h
                distances[i][j] = np.sqrt(dx**2 + dy**2)
    else:
        distances = np.empty((len(letters), 21, 21))
        for i in range(len(letters)):
            w = np.max(arr[i][:, 0]) - np.min(arr[i][:, 0])
            h = np.max(arr[i][:, 1]) - np.min(arr[i][:, 1])
            for j in range(21):
                for k in range(21):
                    dx = (arr[i][k][0] - arr[i][j][0]) / w
                    dy = (arr[i][k][1] - arr[i][j][1]) / h
                    distances[i][j][k] = np.sqrt(dx**2 + dy**2)
    return distances

def ConvertToSample(hand_landmarks) -> np.array:
    coords = list()
    
    for landmark in hand_landmarks.landmark:
        coords.append([landmark.x, landmark.y])
    received = np.array(coords)
    observed = ComputeDistances(received, 2)
    observed = observed.reshape(21*21)
    return observed

def RecognizeLetter(hand_landmarks: np.array, w: int, h: int) -> str:
    coords = list()
    expected = np.load('data/base.npy')
    errors = np.empty(len(letters))
    
    for landmark in hand_landmarks.landmark:
        coords.append([landmark.x * w, landmark.y * h])
    received = np.array(coords)

    observed = ComputeDistances(received, 2)

    for letter in range(len(letters)):
        errors[letter] = np.sum(((expected[letter] - observed) / 21)**2)
    letter = letters[np.argmin(errors)]
    min_error = np.min(errors)
    confidence = (threshold - min_error) / threshold
    # if min_error > 0.003:
        # return '', confidence
    return letter, confidence

def CreateBase():
    capture = cv2.VideoCapture(0)
    img: cv2.Mat
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand = mp_hands.Hands()
    
    coords = list()
    letters = list()
    curr_record = 0
    record = False

    while(True):
        not_empty, img = capture.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        if not_empty:
            results = hand.process(imgRGB)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if record:
                        for point in hand_landmarks.landmark:
                            coords.append([point.x * w, point.y * h])
                        arr = np.array(coords)
                        letters.append(arr)
                        coords.clear()
                        print('Saved letter {} successfully'.format(curr_record))
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Image', img)
        record = False
        key = cv2.waitKey(1)
        if key == 32:
            record = True
            curr_record += 1
        elif key == 114:
            coords.pop()
        elif key == 113:
            break

    letters = np.array(letters)
    distances = ComputeDistances(letters, 3)
    np.save('data/base.npy', distances)
    print('Base created successfully')

def EditBase():
    print('Base edited successfully')

def AddNewSample(sample: np.array, label: str, filepath: str='pjm_testing_set.csv'):
    sample = list([str(_) for _ in sample])
    with open('PJMRecognizer/data/'+filepath, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        if os.path.getsize('./PJMRecognizer/data/{}'.format(filepath)) == 0:
            columns = list()
            columns.append('label')
            for _, data in enumerate(sample):
                columns.append('dist{}'.format(_))
            writer.writerow(columns)
        writer.writerow([label] + sample)

def DeleteLastSample(filepath: str):
    f = open('data/'+filepath, 'w')
    lines = f.readlines()
    lines = lines[:-1]

    writer = csv.writer(f, delimiter=',')
    for line in lines:
        writer.writerow(line)