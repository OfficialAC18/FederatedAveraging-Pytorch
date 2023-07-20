from models.CNN import CNN
from models.MLP import MLP
import torch

def create_model(args, server = False):
    if args.model == "CNN":
        model = CNN(args)
    else:
        model = MLP(args)

    #Initializing the Lazy Params in the models
    if args.dataset in ["MNIST","Fashion-MNIST"]:
        model(torch.rand(size=(1,28*28)))
    else:
        model(torch.rand(size=(1,3,32,32)))    
    
    if server:
        args.total_params = sum(p.numel() for p in model.parameters())
        args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        args.num_classes = 10 if args.dataset != "CIFAR-100" else 100

    return model