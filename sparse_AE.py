import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import logging as log
import os
import matplotlib.pyplot as plt
import math
import sys
import pickle

# global constants
BATCH_SIZE = 256
BETA = 3
RHO = 0.01
N_EPOCHS = 300
use_sparse = False

LENGTH = 200
DIAG = False
CUT_W =200
DIVIDE =  False
SPARSE= True 
LOG = True
MAX = 55994
D1 = 0.0
D2 = 0.5
IN = 1
C1 = 8
C2 = 8
C3 = 8
P1 = 2
L1 = 8000 
OUT = 1000
MASK =torch.Tensor([[0,1,0,0,0],[0,1,1,0,0],[0,1,1,1,0],[0,0,1,1,0],[0,0,0,1,0]])
MASK = 8* MASK[None,None]
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        #self.conv1 = nn.Sequential(
        self.conv1 = nn.Conv2d(IN, C1, 5, stride=1, padding=2)
        #    ,nn.ReLU(inplace=True)
        #)
        self.MP1 = nn.MaxPool2d(P1)
        self.conv2 = nn.Sequential(
            nn.Dropout(D1),
            nn.Conv2d(C1, C2, 5, stride=1,padding=2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(D1),
            nn.Conv2d(C2, C3, 5, stride=1,padding=2),
            nn.ReLU(inplace=True),
        )
        self.lin1 = nn.Sequential(
            nn.Dropout(D2),
            nn.Linear(C3 * 100 *10,OUT),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(OUT, C3 * 100 * 10)
        )
        self.tra1 = nn.Sequential(
            nn.Dropout(D1),
            nn.ConvTranspose2d(C3, C2,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True),
        )
        self.tra2 = nn.Sequential(
            nn.Dropout(D1),
            nn.ConvTranspose2d(C2,C1,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True),
        )
        self.up1 = nn.Upsample(scale_factor=P1, mode="bilinear")

        self.tra3 =  nn.Sequential(
            nn.Dropout(0.1),
            nn.ConvTranspose2d(C1,IN,kernel_size=5,stride=1, padding=2),
            #nn.ReLU(True),
            nn.Sigmoid()
        )
        self.W1 = nn.Parameter(torch.randn(C1,IN,5,5)* MASK)
        self.W2 = nn.Parameter(torch.randn(C2,C1,5,5)* MASK)
        self.W3 = nn.Parameter(torch.randn(C3,C2,5,5)* MASK)
        self.W4 = nn.Parameter(torch.randn(OUT,C3,5,5)* MASK)


        
        #c1 = F.conv2d(x, self.W1 * MASK,stride=1, padding =2)
        #c2 = F.conv2d(p1, self.W2 * MASK,stride=1, padding =2)
        #c3 = F.conv2d(c2, self.W3 * MASK,stride=1, padding =2)
    def forward(self, x):
        log.debug(x.shape)
        #print(self.W.shape)
        #print(self.W[0][0])
        c1 = self.conv1(x)
        print(self.conv1.weight.data[0][0])
        #self.conv1.weight.data = self.conv1.weight.data * MASK
        #c1 = F.conv2d(x, self.W1 ,stride=1, padding =2)
        ("c1",c1.shape)
        p1 = self.MP1(c1)
        log.debug("p1",p1.shape)
        #c2 = self.conv2(p1)
        #log.debug("c2",c2.shape)
        #c3 = self.conv3(c2)
        #log.debug("c3",c3.shape)
        c3 = p1
        c3 = c3.view(c3.size(0), -1)
        l1 = self.lin1(c3)
        encoded = l1
        log.debug("enc",encoded.shape)
        
        
        l2 = self.lin2(encoded)
        l2 = l2.view(l2.size(0),C3, 100 ,10)
        log.debug("l2",l2.shape)
        #t1 = self.tra1(l2)
        #log.debug("t1",t1.shape)
        #t2 = self.tra2(t1)
        #log.debug("t2",t2.shape)
        u1 = self.up1(l2)
        log.debug("u1",u1.shape)
        t3 = self.tra3(u1)
        log.debug("t3",t3.shape)
        decoded = t3
        log.debug("dec",decoded.shape)
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
    rho = torch.FloatTensor([RHO for _ in range(1000)]).unsqueeze(0)
    div = ""
    log = ""
    diag = ""
    if LOG:
        log = "_log"
    if DIVIDE:
        div = "_div"
    if DIAG:
        diag = "_diag"
    train_set = pickle.load( open( "../Data/chr4_200"+log+div+diag+".p", "rb" ) )
    test_set = pickle.load( open( "../Data/chr4_200_test"+log+div+diag+".p", "rb" ) )
    print(len(train_set))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False)

    model = SparseAutoencoder()
    #optimizer = optim.Adam(auto_encoder.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001)
    for epoch in range(N_EPOCHS):
        for b_index, (x) in enumerate(train_loader):
            x = x[0]
            x = Variable(x)
            encoded, decoded = model(x)
            criterion = nn.L1Loss()
            #criterion = nn.MSELoss()
            loss = criterion(decoded,x)
            if use_sparse:
                rho_hat = torch.sum(encoded, dim=0, keepdim=True)
                sparsity_penalty = BETA * kl_divergence(rho, rho_hat)
                train_loss = loss + sparsity_penalty
            else:
                train_loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for b_index, (x) in enumerate(test_loader):
            x = x[0]
            x = Variable(x)
            encoded, decoded = model(x)
            criterion = nn.L1Loss()
            test_loss = criterion(decoded,x)
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                },  "../Data/autoencoder"+str(epoch)+".pt" )
        print("Epoch: [%3d], Loss: %.4f Val: %.4f" %(epoch + 1, train_loss.data,
                                                        test_loss.data))
        sys.stdout.flush()

    torch.save(model.state_dict(), "../Data/autoencoder.pt"  )

if __name__== "__main__":
    train()
