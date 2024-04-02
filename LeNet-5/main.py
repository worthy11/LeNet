from lenet_keras import *
from preprocessing import *

(train_set, train_labels), (test_set, test_labels), classes = LoadData()
model = LeNet5(train_set, train_labels, test_set, test_labels, classes, epochs=0)
model.TrainModel()
capture = cv2.VideoCapture(0)
img: cv2.Mat

while(True):
    _, img = capture.read()
    crop = img[100:400, 400:700]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (28, 28))

    sample = np.array(crop)
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)
    label = model.Predict(sample)
    cv2.putText(img, label, (300, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5) # Display result
    cv2.imshow('Image', img)
    cv2.waitKey(1)