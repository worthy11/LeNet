import numpy as np
from config import *

class Loss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(Loss, self).__init__()
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
                boxes_iou = [iou(predicted[i][j][self.C+b*5+1:self.C+b*5+5], expected[i][j][self.C+1:self.C+5]) for b in range(NUM_BOXES)]
                max_box = max()
                min_box = min(boxes)
                for b in range(NUM_BOXES):
                    if predicted[i][j][b*5] >= OBJ_THRESHOLD and predicted[i][j][b*5] == max_box:
                        error_mid += (expected[i][j][b+1]-predicted[i][j][b*5+1])**2 # x coord
                        error_mid += (expected[i][j][b+2]-predicted[i][j][b*5+2])**2 # y coord
                    
                        error_dim += (np.sqrt(expected[i][j][b+3])-np.sqrt(predicted[i][j][b*5+3]))**2 # width
                        error_dim += (np.sqrt(expected[i][j][b+4])-np.sqrt(predicted[i][j][b*5+4]))**2 # height

                        error_obj += (expected[i][j][b*5]-predicted[i][j][b*5])**2
                    if predicted[i][j][b*5] <= OBJ_THRESHOLD and predicted[i][j][b*5] == min_box:
                        error_noobj += (expected[i][j][b*5]-predicted[i][j][b*5])**2
                
                for c in range(NUM_CLASSES):
                    if 
                    error_class += (expected[i][j][NUM_BOXES*5+c]-predicted[i][j][NUM_BOXES*5+c])**2
                
        loss = LAMBDA_COORD * (error_mid + error_dim) + error_obj + LAMBDA_NOOBJ * error_noobj + error_class