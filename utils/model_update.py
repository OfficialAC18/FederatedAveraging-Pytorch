from utils.model_creator import create_model
import torch.optim
import math
from torch.optim import lr_scheduler
from utils.abs_loss_diff import AbsLossDiff
from torch.utils.data import DataLoader, Sampler
from torch.nn import NLLLoss

def model_update(args,global_state_dict, dataset):
    """
    Function which trains the client model

    :params
    args -> The Model Training Arguments
    global_state_dict -> The state_dict of the global model
    dataset -> The dataset of the client


    return:
    The state_dict of the locally trained model
    """

    local_model = create_model(args=args)
    local_model.load_state_dict(global_state_dict)
    loss_diff = AbsLossDiff()
    loss_fn = NLLLoss()

    # The ceil should ensure a batch size of atleast 1
    batch_size = math.ceil(args.batch_size_frac*args.num_elements)
    
    train_dl = DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(
            params=local_model.parameters(),
            lr = args.lr,
            fused=True
        )
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(
            params=local_model.parameters(),
            lr = args.lr,
            dampening=0.1,
            momentum=0.1,
            nesterov=True
        )
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(
            params=local_model.parameters(),
            lr = args.lr,
            fused=True
        )

    if args.sched == "cyclic":
        scheduler = lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=args.lr,
            max_lr=args.lr * 10
            )
    elif args.sched == "1cycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr * 10,
            epochs=args.client_epochs,
        )
    elif args.sched == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=len(dataset)//batch_size
        )
    elif args.sched == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=20,
            gamma=0.1
        )

    #Training Loop
    local_model.train()
    for epoch in range(args.client_epochs):
        for x,y in train_dl:
            optimizer.zero_grad()
            output = local_model(x)
            loss



    return  local_model.state_dict()