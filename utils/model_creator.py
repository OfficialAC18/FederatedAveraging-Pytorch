from models.CNN import CNN
from models.MLP import MLP

def create_model(args):
    if args.model == "CNN":
        model = CNN(args)
    else:
        model = MLP(args)
        
    args.total_params = sum(p.numel() for p in model.parameters())
    args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model