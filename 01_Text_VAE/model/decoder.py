import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, params):
        super(decoder, self).__init__()
        
        self.params = params
        
        self.embedding = nn.Embedding(self.params.output_dim, self.params.embed_dim)
        
        if params.rnn_type == 'rnn':
            self.rnn = nn.RNN()
        elif params.rnn_type == 'gru':
            self.rnn = nn.GRU()
        elif params.rnn_type == 'lstm':
            self.rnn = nn.LSTM()
        else:
            raise ValueError()
        
        