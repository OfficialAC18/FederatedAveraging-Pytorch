import torch
from collections import OrderedDict

def create_avg_state_dict(device,state_dict):
    """
    Creates a newly initialized state-dict which will be used to compute run averages

    :params
    device -> Device which is used to train the models
    state_dict -> The state-dict which is to be imitated (i.e The global state-dict)

    """

    avg_state_dict = OrderedDict()
    
    #Creating an averaging state dict for each run
    for param_tensor in state_dict:
        avg_state_dict[param_tensor] = torch.zeros(state_dict[param_tensor].size(),device=device)
    

    return avg_state_dict



def update_state_dict(iter, avg_state_dict, iter_state_dict):
    """
    Updates the running average of the weights of each run

    :params
    iter -> The iteration of the current run
    avg_state_dict -> The running average state-dict
    iter_state_dict -> The state-dict of the current iteration
    """

    # Running Averages formula
    # R_avg_n = R_avg_n-1 + 1/n * (R - R_avg_n-1)
    for param_tensor in avg_state_dict:
        avg_state_dict[param_tensor] += (1/iter)*(iter_state_dict[param_tensor] - avg_state_dict[param_tensor]) 

    
    return avg_state_dict

    


