from data_loader import *
from model import *
from functions import *

def main_primitive():
    capture = cv2.VideoCapture(0)
    img: cv2.Mat
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand = mp_hands.Hands()

    coords = list()
    mode = 'R' # C for create, E for edit, R for recognize

    while(True):
        if mode == 'C':
            print('Creating base')
            CreateBase()
            mode = 'R'
        elif mode == 'E':
            print('Editing base')
            EditBase()
            mode = 'R'
        else: 
            not_empty, img = capture.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape

            if not_empty:
                results = hand.process(imgRGB)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        letter, confidence = RecognizeLetter(hand_landmarks, w, h)
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(img, letter, (w // 2, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, confidence*255, (1-confidence)*255), 5)
                    coords.clear()
                    
                cv2.imshow('Image', img)
            key = cv2.waitKey(1)

def main_smart():
    classes: dict[int, str] = {
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
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z'
    }
    recognizer = Model(classes, epochs=10, batch_size=1, learning_rate=0.00001, from_checkpoint=True)

    capture = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hand_recognizer = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    work_mode = 114       # Recognize ('r') by default
                          # Press 't' to retrain the model
                          # Press 'n' to add new samples
                          # Press 'd' to delete last sample
    save_mode = 'testing' # Add samples to testing set by default
                          # Press space to change

    while(True):
        not_empty, img = capture.read()
        if not_empty:
            h, w, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hand_recognizer.process(imgRGB)

            if results.multi_hand_landmarks: # If hand(s) is (are) detected
                for hand_landmarks in results.multi_hand_landmarks:
                    sample = ConvertToSample(hand_landmarks) # Convert landmarks to np.array
                    match work_mode:
                        case 114:
                            # Recognize
                            sample = np.expand_dims(sample, axis=-1)
                            sample = np.expand_dims(sample, axis=0)
                            label, confidence = recognizer.Predict(sample)  # Feed to network
                            color = (0, confidence*255, (1-confidence)*255) # Set color depending on confidence
                            cv2.putText(img, label, (w // 2, 70), cv2.FONT_HERSHEY_PLAIN, 5, color, 5) # Display result
                        
                        case 116:
                            # Retrain model
                            print('Save to checkpoint? [y/n]')
                            choice = input()
                            print(choice)
                            if choice != 'y' and choice != 'n':
                                print('Failed to retrain model')
                            else:
                                if choice == 'n':
                                    choice = False
                                else:
                                    choice = True
                                recognizer.TrainModel(choice)
                                print('Retrained model successfully')
                        
                        case 110:
                            # Add new samples
                            print('Waiting for label')
                            label = str(cv2.waitKey() - 97) # From ASCII table
                            filepath = 'pjm_' + save_mode + '_set.csv'
                            AddNewSample(sample, label, filepath)
                            print('Saved sample with label {}'.format(label))
                        
                        case 100:
                            # Delete last sample
                            filepath = 'pjm_' + save_mode + '_set.csv'
                            DeleteLastSample(filepath)
                            print('Removed sample from {} set'.format(filepath))
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    work_mode = 114
                        
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)

        if key == 32:
            if save_mode == 'testing':
                save_mode = 'training'
                print('Switching to training set')
            else:
                save_mode = 'testing'
                print('Switching to testing set')
        elif key != -1:
            work_mode = key

main_smart()