import argparse

def train_options():

    parser = argparse.ArgumentParser()
    me_group = parser.add_mutually_exclusive_group() #This is for determinining if 

    parser.add_argument("--dataset",
                         type = str,
                         required=True,
                         help = "The dataset on which you would like to train (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100)")
    
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help = "The model which you would like to train (2-Layer MLP or 2-Layer 5x5 CNN)")
    
    parser.add_argument('--iid',
                        type=bool,
                        default=False,
                        help = "Independent and Identical Distributed Random Variables. If the distribution of data at each client is \
                            similar or not")
    
    parser.add_argument('')