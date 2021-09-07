## https://github.com/kefirski/pytorch_RVAE

import argparse
import os

import numpy as np
import torch
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from my_model import My_TextVae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI', help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA', help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR', help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR', help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT', help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE', help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD', help='ce result path (default: '')')
    
    args = parser.parse_args()
    
    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)
    
    myVAE = My_TextVae(parameters)
    
    if args.use_cuda:
        myVAE = myVAE.cuda()
    