from torch import nn
import torch.nn.functional as func
import torch

class LeNet5Torch(nn.Module):
    def __init__(self, trainloader, testloader, epochs=5, batch_size=64, learning_rate=0.00001):
        super(LeNet5Torch, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), bias=True)
        self.C2 = nn.Conv2d(6, 16, (5,5), bias=True)
        self.P = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.FC1 = nn.Linear(in_features=256, out_features=120, bias=True)
        self.FC2 = nn.Linear(120, 84, bias=True)
        self.FC3 = nn.Linear(84, 10, bias=True)
        self.SM = nn.Softmax()

        self.trainloader = trainloader         
        self.testloader = testloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def forward(self, x):
        x = self.C1(x)
        x = self.P(func.tanh(x))
        x = self.C2(x)
        x = self.P(func.tanh(x))
        x = x.view(-1, 256)
        x = self.FC1(func.tanh(x))
        x = self.FC2(func.tanh(x))
        x = self.FC3(func.tanh(x))
        x = self.SM(x)
        return x

def Predict(model, sample):
    output = model(sample)
    output = list(output[0])
    max_prob = max(output)
    label = output.index(max_prob)
    confidence = max_prob / sum(output)
    return label, confidence