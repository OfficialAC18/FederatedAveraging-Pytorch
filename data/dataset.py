import math
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch

#Wrapper over dataset class for inbuilt Pytorch dataset
class ClientDataset(Dataset):
    def __init__(self, dataset, idxs):
        super(ClientDataset, self).__init__()
        self.dataset = dataset
        self.idxs = idxs
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self,idx):
        image, label = self.dataset[self.idxs[idx]]
        return image, label



def iid_sample(args, dataset):
    """
    Function which yields an IID sample on each call
  
    :params:
    :args -> The arguments of the run
    :dataset -> The dataset which is to be sampled
    """
    
    #This section is for when we are executing only a single run
    if args.fraction:
        num_clients = round(args.fraction * args.num_clients)

        #Checking if each client gets atleast 2 elements
        assert args.num_elements >= 2, "Each client must have atleast 2 examples, currently, there are too many clients. Please rectify."

        #creating a list of idxs to randomly sample from (without replacement)
        all_idxs = np.arange(len(dataset))


        #creating RNG object for sampling
        rng = np.random.default_rng(seed = args.seed)
        
        for _ in range(num_clients):
            sample = rng.choice(a = all_idxs, size = args.num_elements)
            all_idxs = np.setdiff1d(all_idxs, sample, assume_unique=True)
            yield sample
        
    #This section is for when we are executing multiple runs at various fractions of cleints
    else:
        total_runs = math.floor(1/args.stepsize)
        for run in range(total_runs):
            num_clients = round(args.stepsize*run*args.num_clients)

            #Checking if the number of clients is valid
            assert num_clients <= args.num_clients,f"The fraction {args.stepsize*run} of clients is greater than total number of clients, Please rectify."
        
            #Checking if each client gets atleast 2 elements
            assert args.num_elements >= 2, f"At fraction {args.stepsize*run}, each client cannot have a minimum of 2 examples. Please rectify."

            #creating a list of idxs to randomly sample from (without replacement)
            all_idxs = np.arange(len(dataset))

            #creating RNG object for sampling
            rng = np.random.default_rng(seed = args.seed)

            for _ in range(num_clients):
                sample = rng.choice(a = all_idxs, size = args.num_elements)
                all_idxs = np.setdiff1d(all_idxs, sample, assume_unique=True)
                yield sample


def non_iid_sample(args, dataset):
    """
    Function which yields a Non-IID sample on each call
    
    :params:
    :args -> The arguments of the run
    :dataset -> The dataset which is to be sampled
    """

    #This section is for when we are executing only a single run
    if args.fraction:
        num_clients = round(args.fraction * args.num_clients)

        #Checking if each client gets atleast 2 elements
        assert args.num_elements >= 2, "Each client must have atleast 2 examples, currently, there are too many clients. Please rectify."

        #Get a Tensor of sorted indices and sample incrementally
        all_idxs = torch.Tensor(dataset.targets).sort()[1]
     
        for i in range(num_clients):
            #Sampling examples in a linear fashion
            sample =  all_idxs[(i*args.num_elements):((i+1)*args.num_elements)]
            all_idxs = all_idxs[(all_idxs[:, None] != sample).all(dim=1)]
            yield sample
        
    #This section is for when we are executing multiple runs at various fractions of cleints
    else:

        total_runs = math.floor(1/args.stepsize)
        for run in range(total_runs):
            num_clients = round(args.stepsize*run*args.num_clients)

            #Checking if the number of clients is valid
            assert num_clients <= args.num_clients,f"The fraction {args.stepsize*run} of clients is greater than total number of clients, Please rectify."
            

            #Checking if each client gets atleast 2 elements
            assert args.num_elements >= 2, f"At fraction {args.stepsize*run}, each client cannot have a minimum of 2 examples. Please rectify."

            #Get a Tensor of sorted indices and sample incrementally
            all_idxs = torch.Tensor(dataset.targets).sort()[1]

            for i in range(num_clients):
                sample = all_idxs[(i*args.num_elements):((i+1)*args.num_elements)]
                all_idxs = all_idxs[(all_idxs[:,None] != sample).all(dim=1)]
                yield sample



def __create_client_dataset(args,dataset):

    if args.iid:
        sample_gen = iid_sample(args=args, dataset=dataset)
    else:
        sample_gen = non_iid_sample(args=args,dataset=dataset)

    while True:
        try:
            idxs = next(sample_gen)
            yield ClientDataset(dataset = dataset, idxs = idxs)

        except StopIteration:

            #This is to signify that the dataset has been exhausted
            return -1

def create_client_dataset(args):
    if args.dataset == "MNIST":
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081))
            ])
        )
    elif args.dataset == "Fashion-MNIST":
        dataset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    elif args.dataset == "CIFAR-10":
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))])
        )
    elif args.dataset == "CIFAR-100":
        dataset = datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))])
    
        )   
    
    #To Calculate total number of files
    args.num_elements = math.floor(len(dataset)/args.num_clients)
    return __create_client_dataset(args,dataset)