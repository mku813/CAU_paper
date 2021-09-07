## https://github.com/kefirski/pytorch_RVAE

import torch
import torch.nn as nn
import torch.nn.functional as F


class My_TextVae(nn.modules):
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        