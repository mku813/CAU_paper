import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from cold_posterior.VAE import model

vae = model.VAE(x_dim= 28*28, h_dim1=512, h_dim2=256, z_dim=2)

if torch.cuda.is_available():
    vae.cuda()

vae