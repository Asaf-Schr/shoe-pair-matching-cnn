import numpy as np
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 32
        kernel_size = 5
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        # TODO: complete this method
        # layer 2: n -> 2n
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        # layer 3: 2n -> 4n
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, padding=padding)
        # layer 4: 4n -> 8n
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # the input image size is 448x224.
        # we pass through 4 pooling layers , each dividing dimensions by 2.
        # total reduction factor = 2^4 = 16.
        # final Height = 448 / 16 = 28
        # final Width  = 224 / 16 = 14
        # final Channels = Output of conv4 = 8*n
        
        flatten_size = (8 * n) * (448 // 16) * (224 // 16)

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=2) # same pair vs different pair

    def forward(self, inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        out = self.conv1(inp)
        # TODO: complete this function
        #B1
        out = self.relu(out)
        out = self.pool(out)

        #B2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        #B3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)

        #B4
        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        n = 32
        kernel_size = 5
        padding = (kernel_size-1)//2

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=kernel_size, padding=padding)
        
        # rest of the convolutional
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)
        
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # New Input Height: 224 (after splitting the 448 image)
        # New Input Width: 224
        # Total reduction factor: 16 (2^4)
        # Final Height: 224 / 16 = 14
        # Final Width:  224 / 16 = 14
        
        flatten_size = (8 * n) * (224 // 16) * (224 // 16)
        
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=2)

    # TODO: complete this method
    def forward(self, inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        # TODO: complete this function
        shoe1 = inp[:, :, :224, :] 
        shoe2 = inp[:, :, 224:, :]
        
        out = torch.cat((shoe1, shoe2), dim=1) #shape: (N, 6, 224, 224)
        #B1
        out = self.conv1(out)
        out = self.relu(out)
        out = self.pool(out)

        #B2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        #B3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)

        #B4
        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)



        return out
