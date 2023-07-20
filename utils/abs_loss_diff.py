import math
import torch
class AbsLossDiff:
    def __init__(self,args) -> None:
        self.last_loss = torch.tensor(math.inf, device = args.device)
        self.loss_diff = torch.tensor(math.inf, device = args.device)

    def log_loss(self,loss):
        """
        Keeps track of the average loss per epoch
        """


    def calc_error_diff(self,loss):
        self.loss_diff = torch.abs(
            self.last_loss - loss
        )
        self.last_loss = loss
