import logging

#Setting the config for the logger
logging.basicConfig(format='%(levelname)s-%(asctime)s: %(message)s')

#Setting the default logger level
logging.getLogger().setLevel(logging.INFO)

#To print the number in a more readable format
#Link:https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def model_info(args):
    """
    Print out the details of the model and the current run

    :params
    args -> The args for the training run
    mode -> The model to be used in the run (Used to calculate parameters)
    """

    logging.info("*********RUN INFO**********")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model: {args.model} ({args.num_classes} classes)")
    logging.info(f"Number of Parameters: {human_format(args.total_params)}")
    logging.info(f"Number of trainable params: {human_format(args.trainable_params)}")
    logging.info(f"IID: {str(args.iid)}")
    logging.info(f"Fraction: {args.fraction}" if args.fraction
                  else f"Stepsize: {args.stepsize}")
    logging.info(f"Total Number of Clients: {args.num_clients}")
    
    if args.fraction:
        logging.info(f"Total Number of Clients (Per Run): {args.num_clients_round}")
        logging.info(f"Number of files per client: {args.num_elements}")
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Optimizer: {args.optim}")
    logging.info(f"GPU: {str(args.gpu)}")
    logging.info(f"Total Number of Epochs (at server): {args.epochs}")
    logging.info(f"Save Models: {str(args.save_models)}")