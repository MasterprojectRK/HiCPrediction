import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import math
import pickle


# global constants
BATCH_SIZE = 32
BETA = 3
RHO = 0.01
N_INP = 3810
N_HIDDEN = 1000
N_EPOCHS = 100
use_sparse = False

class SparseAutoencoder(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inp)

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded




def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2

def train():
    rho = torch.FloatTensor([RHO for _ in range(N_HIDDEN)]).unsqueeze(0)

    train_set = pickle.load( open( "../Data/chr4_100kb.p", "rb" ) )
    print(train_set[0])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)

#test_loader = torch.utils.data.DataLoader(
#        dataset=test_set,
#        batch_size=BATCH_SIZE,
#        shuffle=False)

    auto_encoder = SparseAutoencoder(N_INP, N_HIDDEN)
    optimizer = optim.Adam(auto_encoder.parameters(), lr=0.001)
    for epoch in range(N_EPOCHS):
        print(len(train_loader))
        for b_index, (x) in enumerate(train_loader):
            x = x[0]
            x = Variable(x)
            encoded, decoded = auto_encoder(x)
            MSE_loss = (x - decoded) ** 2
            MSE_loss = MSE_loss.view(1, -1).sum(1) / BATCH_SIZE
            if use_sparse:
                rho_hat = torch.sum(encoded, dim=0, keepdim=True)
                sparsity_penalty = BETA * kl_divergence(rho, rho_hat)
                loss = MSE_loss + sparsity_penalty
            else:
                loss = MSE_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: [%3d], Loss: %.4f" %(epoch + 1, loss.data))

    torch.save(auto_encoder.state_dict(), "../Data/autoencoder.pt"  )

if __name__== "__main__":
    train()
