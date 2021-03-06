import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        
        self.params = params
        
        if params.rnn_type == 'rnn':
            self.rnn = nn.RNN()
        elif params.rnn_type == 'gru':
            self.rnn = nn.GRU()
        elif params.rnn_type == 'lstm':
            self.rnn = nn.LSTM()
        else:
            raise ValueError()
        
        self.embedding = nn.Embedding(self.params.input_dim, self.params.embed_dim)
        
    def forward(self, input):
        
        embed = self.embedding(input).view(1,1,-1)
        output, hidden = self.rnn(embed)
        
        return output, hidden
    
    






