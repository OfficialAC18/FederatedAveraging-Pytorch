#### TO-DO LIST #### 
# 1. Implement Multiple Optimizers, with Schedulers and such (e.g. Cosine Scheduler for SGD)
# 2. Implement a general Dataset class to inherit from (Done)
# 2a. Implement IID sampling (Done)
# 2b. Implement Non-IID Sampling (Done)
# 3. Implement the class for each Dataset (Done)
# 4. Create Both Models (Done)
# 5. Create the training loop for the local models (Done)
# 6. Implement MeanSquaredLoss to test when to stop the training of the local epochs
# 7. Create a model info function
# 8. Create asserts to ensure that each client gets atleast 2 examples (Don't let the program run otherwise) (Done)
# 9. Implement unequal sampling

from utils.options_parser import train_options
import torch
import random
import logging
from utils.model_creator import create_model
from data.dataset import create_client_dataset
from utils.model_update import model_update
from utils.run_average import create_avg_state_dict, update_state_dict

#Setting the config for the logger
logging.basicConfig(format='%(levelname)s-%(asctime)s-%(message)s')

#Attain the training criteria
args = train_options()


if args.gpu == True:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logging.info("GPU not available, Switching to CPU")
        device = torch.device('cpu')

else:
    device = torch.device('cpu')

#Create the global_model based on args and load it
#on relevant hardware
global_model = create_model(args)
global_model.to(device)

#State-dict of the global model
global_state_dict = global_model.state_dict() 

#Generator for the client datasets
data_gen = create_client_dataset(args)

#dict for holding the dataset of each client
client_data = {}

#Calculating the number of clients per round
args.num_clients_round = round(args.fraction * args.num_clients)
#Checking if the number of clients is valid
assert args.num_clients_rounds <= args.num_clients, f"The fraction of clients is higher than the total number of clients. Please rectify."

#Implementing FedAveraging
for _ in range(args.epoch):
    #Sampling a random set of clients
    clients_in_round = random.samples(range(args.num_clients),args.num_clients_round)

    #TODO: Turn this into weighted average maybe??
    running_avg = create_avg_state_dict(device=device,state_dict=global_state_dict)
    for idx,c in enumerate(clients_in_round):
        if c in client_data.keys():
            local_state_dict = model_update(args,global_state_dict=global_state_dict,dataset=client_data[c])
        else:
            client_data[c] = next(data_gen)
            local_state_dict = model_update(args,global_state_dict=global_state_dict,dataset=client_data[c])

        #Update the average weights
        running_avg = update_state_dict(iter=idx, avg_state_dict=running_avg,iter_state_dict=local_state_dict)
    
    #Updating the weights of the global model
    global_state_dict = running_avg








