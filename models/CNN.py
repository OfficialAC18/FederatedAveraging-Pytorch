from torch import nn 
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        if args.dataset in ["MNIST", "Fashion-MNIST"]:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                                   kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                                   kernel_size=5, stride=1)
            self.maxpool = nn.MaxPool1d(kernel_size=2)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                                   kernel_size=5, stride=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                                   kernel_size=5, stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=2)

        #The stateless layers are common for both    
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        
        self.linear = nn.LazyLinear(out_features=512)

        if args.dataset in ["MNIST","Fashion-MNIST","CIFAR-10"]:
            self.head = nn.Linear(in_features=512, out_features=10)   
        else:    
            self.head = nn.Linear(in_features=512, out_features=100) 

    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        x = F.softmax(self.head(x),dim=-1)

        return x
    
