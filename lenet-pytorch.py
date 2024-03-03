from torch import nn
import torch.nn.functional as func
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms

batch_size = 32
epochs = 10
learn_rate = 0.001
momentum = 0.9

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))]
     )
trainset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), bias=True)
        self.C2 = nn.Conv2d(6, 16, (5,5), bias=True)
        self.P = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.FC1 = nn.Linear(in_features=256, out_features=120, bias=True)
        self.FC2 = nn.Linear(120, 84, bias=True)
        self.FC3 = nn.Linear(84, 10, bias=True)
    
    def forward(self, x):
        x = self.C1(x)
        x = self.P(func.tanh(x))
        x = self.C2(x)
        x = self.P(func.tanh(x))
        x = x.view(-1, 256)
        x = self.FC1(func.tanh(x))
        x = self.FC2(func.tanh(x))
        x = self.FC3(func.tanh(x))
        return x

lenet = Lenet()
optimizer = torch.optim.SGD(lenet.parameters(), learn_rate, momentum)
loss_func = nn.CrossEntropyLoss()
def train(epochs):
    if __name__=='__main__':
        for epoch in range(epochs):    
            epoch_loss = 100.
            batch_loss = 0.

            lenet.train(True)
            for idx, data in enumerate(trainloader):
                xs, labels = data
                
                optimizer.zero_grad() # Reset gradient for new batch
                outs = lenet(xs)      # Get outputs
                loss = loss_func(outs, labels) # Compute loss
                loss.backward()       # Compute loss gradients
                optimizer.step()      # Adjust the weights

                batch_loss += loss.item()
                if idx % 100 == 99: # Average loss per 100 batches 
                    batch_loss /= 100
                    print('Avg loss in batches {}-{}: {}'.format(idx-99, idx+1, batch_loss))
                    if batch_loss < epoch_loss:
                        epoch_loss = batch_loss
                    batch_loss = 0.
            
            lenet.eval()
            with torch.no_grad():
                recall = 0.
                for idx, data in enumerate(testloader):
                    xs, labels = data

                    outs = lenet(xs)
                    for idx, out in enumerate(outs):
                        out = list(out)
                        loss = loss_func(outs, labels)
                        max_prob = max(out)
                        max_idx = out.index(max_prob)
                        recall += (max_idx == labels[idx])
                print('Recall in epoch {}: {}'.format(epoch+1, recall/(len(testloader)*batch_size)))

train(10)