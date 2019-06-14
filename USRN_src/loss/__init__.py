import torch
import torch.nn as nn
from importlib import import_module


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        print('Preparing loss function...')

        self.num_gpu = config.num_gpu
        self.losses = []
        self.loss_module = nn.ModuleList()

        if config.model_type == 'normal':
            module = import_module('loss.mse')
            loss_function = getattr(module, 'MSE')()
        if config.model_type == 'usrn':
            module = import_module('loss.mse_var')
            loss_function = getattr(module, 'MSE_VAR')()
        self.losses.append({'function': loss_function})

        self.loss_module.to(config.device)
        if not config.cpu and config.num_gpu > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(self.num_gpu))

    def forward(self, results, label):
        losses = []
        for i, l in enumerate(self.losses):
            if l['function'] is not None:
                loss = l['function'](results, label)
                effective_loss = loss
                losses.append(effective_loss)

        loss_sum = sum(losses)
        if len(self.losses) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum
