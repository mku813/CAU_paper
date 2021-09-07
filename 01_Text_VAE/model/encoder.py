import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, params, rnn='lstm'):
        super(Encoder, self).__init__()
        
        self.params = params
        
        if rnn == 'rnn':
            self.rnn = nn.RNN()
        elif rnn == 'gru':
            self.rnn = nn.GRU()
        elif rnn == 'lstm':
            self.rnn = nn.LSTM()
        
        







