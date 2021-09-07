## https://github.com/kefirski/pytorch_RVAE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .encoder import Encoder
from .decoder import Decoder

class My_TextVae(nn.modules):
    def __init__(self, params):
        super(My_TextVae, self).__init__()
        
        self.params = params
        
        self.encoder = Encoder(self.params)
        
        self.context_mu = nn.Linear(self.params.encoder_encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_log_var = nn.Linear(self.params.encoder_encoder_rnn_size * 2, self.params.latent_variable_size)
        
        self.decoder = Decoder(self.params)
        
    def forward(self, drop_prob
                , encoder_word_input=None, encoder_character_input=None
                , decoder_word_input=None, decoder_character_input=None
                , z=None, init_state=None):
        
        input_length = source.size(0)
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param z: context if sampling is performing
        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        
        if z is None:
            [batch_size, _] = encoder_word_input.size()
            
            encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            
            context = self.encoder(encoder_input)
            
            mu = self.context_mu(context)
            logvar = self.context_log_var(context)
            std = t.exp(0.5 * logvar)
            
            z = Variable(torch.randn([batch_size, self.params.latent_variable_size]))
            
            z = z * std + mu
            
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu,2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None
        
        