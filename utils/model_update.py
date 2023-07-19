from utils.model_creator import create_model

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






    return  local_model.state_dict()