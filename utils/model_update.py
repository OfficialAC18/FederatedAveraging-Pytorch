from utils.model_creator import create_model
import torch.optim 

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

    if args.optim == "Adam":
        optim = torch.optim.Adam(
            params=local_model.parameters(),
            lr = args.lr,
            fused=True
        )
    elif args.optim == "SGD":
        optim = torch.optim.SGD(
            params=local_model.parameters(),
            lr = args.lr,
            dampening=0.1,
            momentum=0.1,
            nesterov=True
        )
    elif args.optim == "AdamW":
        optim = torch.optim.AdamW(
            params=local_model.parameters(),
            lr = args.lr,
            fused=True
        )

    


    return  local_model.state_dict()