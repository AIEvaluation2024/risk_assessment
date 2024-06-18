# implement the neural network architecture
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN_input2_output1(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,1)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
    #    out = F.relu(out)
    #    out = self.linear5(out)
    #    out = F.relu(out)
    #    out = self.linear6(out)

        return out

class NN_input2_output2(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
     #   self.linear3 = nn.Linear(64,64)
     #   self.linear4 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,2)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
      #  out = F.relu(out)
      #  out = self.linear5(out)
      #  out = F.relu(out)
      #  out = self.linear6(out)

        return out



#architecture can be used for california housing data 
class NN_input8_output1(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 8
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,1)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out

class NN_input8_output2(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 8
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,2)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out


# A deeper NN designed for 16 inputs and 2 output
class NN_input16_output2_deeper(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 16
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,16)
        self.linear5 = nn.Linear(16,2)


    def forward(self,image):
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)

        return out


#architecture can be used for CCPP dataset
class NN_input4_output1(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 4
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,1)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
 
        return out

#architecture can be used for CCPP dataset
class NN_input4_output2(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 4
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,2)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
 
        return out


class NN_input16_output1(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 16
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,1)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out

class NN_input16_output2(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 16
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,2)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out


class NN_input11_output1(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 11
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,1)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out

class NN_input11_output2(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 11
        self.linear1 = nn.Linear(input_size,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,2)


    def forward(self,image):
        # flatten the list of images
        out = image.view(image.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out


