from utils.model_creator import create_model
import torch.optim
import math
from torch.optim import lr_scheduler
from utils.abs_loss_diff import AbsLossDiff
from torch.utils.data import DataLoader, Sampler
from torch.nn import CrossEntropyLoss

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
    local_model.to(args.device)

    loss_diff = AbsLossDiff()
    loss_fn = CrossEntropyLoss()

    # The batch size must be smaller than the dataset
    assert args.batch_size_frac <= 1, f"The batch size is greater than the number of elements, please fix the value for batch size fraction."
    # The ceil should ensure a batch size of atleast 1
    batch_size = math.ceil(args.batch_size_frac*args.num_elements)
    
    train_dl = DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    #Choice of Optimizer
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(
            params=local_model.parameters(),
            lr = args.lr,
            fused=True if args.gpu else False
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
            fused=True if args.gpu else False
        )

    #Choice of Scheduler
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
            steps_per_epoch=len(train_dl)
        )
    elif args.sched == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=batch_size
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
            x,y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            output = local_model(x)

            loss = loss_fn(output, y, reduction='mean')
            loss.backward()

            optimizer.step()

            if args.sched != "step":
                scheduler.step()

        if args.sched == "step":
            scheduler.step()

    return  local_model.state_dict()