from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=200)
        if args.dataset in ["MNIST","Fashion-MNIST","CIFAR-10"]:
            self.head = nn.Linear(in_features=200, out_features=10)
        else:
            self.head = nn.Linear(in_features=200, out_features=100)
    
    def forward(self,x):
        x = self.flatten(x)
        x = F.relu(F.dropout(self.linear1(x)))
        x = self.linear2(x)
        x = F.softmax(self.head(x),dim=-1)
