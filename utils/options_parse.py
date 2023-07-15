import argparse

def train_options():

    parser = argparse.ArgumentParser()

    #This is for determinining if we are going to have multiple runs, with a constant stepsize for C from [0,1] or 
    #if are only going to have a single run
    me_group = parser.add_mutually_exclusive_group() 

    parser.add_argument("--dataset",
                         choices=["MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100"],
                         required=True,
                         help = "The dataset on which you would like to train (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100)")
    
    parser.add_argument("--model",
                        choices=['MLP', 'CNN'],
                        required=True,
                        help = "The model which you would like to train (MLP, CNN)")
    
    parser.add_argument('--iid',
                        type=bool,
                        default=False,
                        help = "Independent and Identical Distributed Random Variables. If the distribution of data at each client is \
                            similar or not (Default set to False)") 
    
    me_group.add_argument("--fraction",
                          type=float,
                          help="The fraction of clients (C) to be selected at each run")
    
    me_group.add_argument("--stepsize",
                          type=float,
                          help="The stepsize (S) that shall be used for the multiple run (0<S<1)")
    
    parser.add_argument('--num_clients',
                        type = int,
                        default=100,
                        help = 'The total number of clients participating in the training (Default set to 100)')
    
    parser.add_argument('--batch_size_frac',
                        type = float,
                        default = 0.1,
                        help = "The percentage of data that is collected in a single batch, if perc. is not a whole number, \
                            rounded down to the nearest whole number (Default set to 0.1)")
    
    parser.add_argument('--verbose',
                        action='store_true')
    
    parser.add_argument('-examples-dist',
                        choices=['equal', 'unequal'],
                        default='equal',
                        help="If the data is to be divided equally amongst the different clients (Default set to equal)")
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help="The learning rate that will be used for the models (Default is 0.01)")
    
    parser.add_argument('--schedule',
                        action="store_true",
                        help="Keep the learning rate static or use a Cosine Schedule while training at client")
    
    parser.add_argument('--optim',
                        choices=["Adam",'SGD','AdamW'],
                        default="Adam",
                        help="Choice of optimizer to use in training at the client end. (Default set to Adam)")
    
    parser.add_argument('--gpu',
                        action="store_true",
                        help="Use the GPU while training at client end")
    
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Set the seed for the run")
    
    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help="Total number of epochs to run at server end")
    
    parser.add_argument('--save-models',
                        action="store_true",
                        help="Store the trained central models, models are stores in the trained models directory")
    
    args = parser.parse_args()
    return args