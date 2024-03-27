import numpy as np
from config import *

def IOU(x1, y1, w1, h1, x2, y2, w2, h2):
    if x1 < x2:
        x_l = max(x1 - w1/2, x2 - w2/2)
        x_r = min(x1 + w1/2, x2 + w2/2)
        y_t = min(y1 - h1/2, y2 - h2/2)
        y_b = max(y1 + h1/2, y2 + h2/2)

    i = (x_r - x_l) * (y_t - y_b)
    u = (w1 * h1) + (w2 * h2) - i
    return i / u

def SE(a, b):
    return (a-b)**2

class Loss():
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        
    def compute(self, expected, predicted):
        # OUPUT SHAPE: 7 x 7 x 30
        # row x column x (c1-20, b1, x, y, w, h, b2, x, y, w, d)
        predicted = predicted.reshape(-1, self.S, self.S, self.C+self.B*5)
        error_mid = error_dim = error_obj = error_noobj = error_class = 0
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                boxes_iou = [IOU(predicted[i][j][self.C+b*5+1:self.C+b*5+5], expected[i][j][self.C+1:self.C+5]) for b in range(NUM_BOXES)]
                max_box = np.argmax(boxes_iou)
                min_box = np.argmin(boxes_iou)
                
                for b in range(NUM_BOXES):
                    # if there is an object in the cell and box has highest iou
                    if expected[i][j][self.C] == 1 and b == max_box:
                        error_mid += SE(expected[i][j][self.C+b+1], predicted[i][j][self.C+b*5+1]) # x coord
                        error_mid += SE(expected[i][j][self.C+b+2], predicted[i][j][self.C+b*5+2]) # y coord
                    
                        # width and height might be negative, so abs. dont divide by 0 in derivative of sqr, so +1e-6.
                        # sign of gradient must stay correct, so sign().
                        sign = (np.sign(predicted[i][j][self.C+b*5+3]), np.sign(predicted[i][j][self.C+b*5+4]))
                        
                        error_dim += SE(np.sqrt(expected[i][j][self.C+b+3]), np.sqrt(predicted[i][j][self.C+b*5+3])) # width
                        error_dim += SE(np.sqrt(expected[i][j][self.C+b+4]), np.sqrt(predicted[i][j][self.C+b*5+4]))**2 # height

                        error_obj += SE(expected[i][j][self.C+b*5], predicted[i][j][self.C+b*5]) # if object present

                    # if there is no object in the cell and box has lowest iou
                    if expected[i][j][self.C] == 0 and b == min_box:
                        error_noobj += SE(expected[i][j][self.C+b*5], predicted[i][j][self.C+b*5]) # if object not present
                
                for c in range(NUM_CLASSES):
                    # if there is an object in the cell
                    if expected[i][j][self.C] == 1:
                        error_class += SE(expected[i][j][c], predicted[i][j][c]) # particular class assessment
                
        loss = LAMBDA_COORD * (error_mid + error_dim) + error_obj + LAMBDA_NOOBJ * error_noobj + error_class