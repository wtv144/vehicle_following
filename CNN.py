import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 5, 2) #input channels is 8 for the number of steps, output is 8? #padding to 0 since input is generally small 
        #think in terms of a spatial component and temporal component 
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84) # maybe replace with nonlinearity
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, obs_seq, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.features = nn.Sequential(
       nn.Conv1d(in_channels = obs_seq, out_channels = 40, kernel_size = 3, padding = 2), #in_channels, out_channels, kernel_size. padding is 0 by default. n_steps input channels. How to determine output channels
        # after inputing [batch_size, 8, 2]. Make padding 1 since assuming 8 x 2, then will be [c_out x 1]
        #so if padding is 0, make kernel_size 1, if kernel_size is 2, makke padding 1
        #with padding 2, output shape iwll be (batch_size, c_out- 40, 4)
        #making large padding to increase dimensionality of last dimension. Want it to go from 2 to 5 in the end result
        nn.BatchNorm1d(40, affine = False),
        nn.ReLU(inplace = True), 
        nn.MaxPool1d(kernel_size = 4, padding = 1, dilation = 1, stride = 1), #current shape is now (batch_size, 40, 3)
        nn.Conv1d(in_channels=40, out_channels = 60, kernel_size = 6, padding = 4), #shape is now (batch_size, 60, 6)
        nn.BatchNorm1d(60, affine = False),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size = 6, padding = 3, stride = 2,dilation=1), #shape is not (batch_size, 60, 4)        
        )
        
        self.regress = nn.Sequential(
        nn.Linear(in_features = 240, out_features =180),
        nn.BatchNorm1d(180, affine = False),
        nn.ReLU(inplace =True),
        nn.Linear(in_features = 180, out_features = 120),
        nn.BatchNorm1d(120, affine = False),
        nn.Tanh(),
        nn.Linear(in_features = 120, out_features = pred_len*5)) #will return 40 features for assume pred_len < 24
        

            
            
    def forward(self, x):
        x = self.features(x)
        #flatten it to remove bugs
        x = torch.flatten(x,1) #remove all except the batch size 
        x = self.regress(x)
        x = x.view(-1,self.pred_len,5)
        return x