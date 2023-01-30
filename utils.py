import torch
import torch.nn as nn
import numpy as np
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def _log_stats(losses, names, num_steps, writer):
        tag_value = {}
        for i in range(len(losses)):
            tag_value[f'loss {names[i]}'] = losses[i]

        for tag, value in tag_value.items():
            writer.add_scalar(tag, value, num_steps)

def set_manual_seed(manual_seed):
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    weight_multiply = 1.5
    with torch.no_grad():
        if classname.find('Conv') != -1:
            m.weight = nn.Parameter(torch.nn.init.kaiming_normal_(m.weight)*weight_multiply)
        elif classname.find('Linear') != -1:
            m.weight = nn.Parameter(torch.nn.init.kaiming_normal_(m.weight)*weight_multiply)
        elif classname.find('BatchNorm2d') != -1:
            m.weight = nn.Parameter(torch.nn.init.normal_(m.weight, 1.0, 0.02)*weight_multiply)
            m.bias = nn.Parameter(torch.nn.init.constant_(m.bias, 0)*weight_multiply)

def init_weights(network, type='normal'):
    with torch.no_grad():
        if type == 'normal':
            network.apply(weights_init_normal)
        elif type == 'kaiming':
            network.apply(weights_init_kaiming)
        else:
            raise NotImplementedError(f'init method "{type}" not implemented')
