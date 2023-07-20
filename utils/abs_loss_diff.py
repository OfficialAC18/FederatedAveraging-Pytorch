import math
import torch
class AbsLossDiff:
    def __init__(self,args) -> None:
        self.last_loss = torch.tensor(math.inf, device = args.device)
        self.loss_diff = torch.tensor(math.inf, device = args.device)

    def calc_error(self,loss):
        self.loss_diff = torch.abs(
            self.last_loss - loss
        )
        self.last_loss = loss
