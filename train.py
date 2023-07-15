#### TO-DO LIST #### 
# 1. Implement Cosine Scheduler for SGD
# 2. Implement a general Dataset class to inherit from
# 3. Implement the class for each Dataset
# 4. Create Both Models (Done)
# 5. Create the training loop for the local models
# 6. Implement MeanSquaredLoss to test when to stop the training of the local epochs
# 7. Create a model info function

from utils.options_parser import train_options
import torch
import logging
from utils.model_creator import create_model

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







