## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs


        # input is torch.Size([1, 224, 224]) torch.Size([68, 2])
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.conv1_bn = nn.BatchNorm2d(32)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)



        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        self.conv2 = nn.Conv2d(32, 40, 5)

        #self.pool = nn.MaxPool2d(2, 2)

        self.conv2_bn = nn.BatchNorm2d(40)


        self.conv3 = nn.Conv2d(40, 64, 4)

        self.conv3_bn = nn.BatchNorm2d(64)


        self.conv4 = nn.Conv2d(64, 128, 3)

        self.conv4_bn = nn.BatchNorm2d(128)

        self.conv4_drop = nn.Dropout(p=0.3)


        #self.fc1 = nn.Linear(25*25*64, 3000)
        self.fc1 = nn.Linear(11*11*128, 5000)

        # dropout with p=0.5
        self.fc1_drop = nn.Dropout(p=0.3)

        #self.fc1_bn = nn.BatchNorm1d(5000)


        self.fc2 = nn.Linear(5000, 136)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv1_bn(x)
        out1 = F.relu(x)


        x = self.conv2(out1)
        x = self.pool(x)
        x = self.conv2_bn(x)
        out2 = F.relu(x)
        #x = self.drop(x)

        x = self.conv3(out2)
        x = self.pool(x)
        x = self.conv3_bn(x)
        out3 = F.relu(x)
        #x = self.drop(x)
        #25x25x64

        x = self.conv4(out3)
        x = self.pool(x)
        x = self.conv4_bn(x)
        out4 = F.relu(x)
        #x = self.drop(x)
        x = self.conv4_drop(out4)
        #11x11x128

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x,out1,out2,out3,out4
