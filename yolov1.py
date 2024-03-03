import torch
import torch.nn as nn
import config

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.arc = config.architecture
        self.in_channels = in_channels
        self.darknet = self.initialize_conv_layers(self.arc)
        self.fc = self.initialize_fc_layers(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
    
    def initialize_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(Layer(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                ))

            elif type(x) == str:
                pass

