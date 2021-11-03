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

optimizer = optim.Adam(vae.parameters())

# return reconstruction error + KL divergence losses

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch, vae, train_loader):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item() / len(data)
            ))
        print("==> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss/len(train_loader.dataset)))

def test(vae):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
    
    test_loss /= len(test_loader.dataset)
    print("==> Test sets loss: {.4f}".format(test_loss))