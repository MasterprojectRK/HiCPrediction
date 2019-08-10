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
import numpy as np
import time

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
# global constants
CHR = "4"
BIN = "20"
DATA_D = "Data2e/"
CHROM_D = DATA_D +BIN+ "Chroms/"
SET_D = DATA_D + BIN +"Sets/"
PRED_D = DATA_D + "Predictions/"
MODEL_D  = DATA_D + "Models/"
CHUNK_D = DATA_D +BIN +"Chunks/"
TEST_D =  DATA_D +BIN+ "Test/"
IMAGE_D = DATA_D +  "Images/"
ORIG_D = DATA_D +  "Orig/"
RHO = 0.1
use_sparse = False

LENGTH = 100
LEN2 = 50
LEN = 2500
DIAG = False
CUT_W =200
DIVIDE =  False
AVG = False
SPARSE= False
LOG = True
LOGNORM = False
MAX = 4846
D1 = 0.2
D2 = 0.1
IN = 1
C1 = 4
C2 = 4
C3 = 8
P1 = 2
L1 = 1000
L2 = 6000
L3 = 3000
OUT = 1000
class SparseAutoencoder(nn.Module):
    def __init__(self,args,c1=C1, c2=C2, c3=C3, p1=P1, l1=L1, out=OUT,
                 d1=D1,d2=D2, inp=IN ):
        super(SparseAutoencoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, c1, 5, stride=1, padding=2)
            ,nn.ReLU(inplace=True)
        )
        self.MP1 = nn.MaxPool2d(p1)
        self.conv2 = nn.Sequential(
            nn.Dropout(d1),
            nn.Conv2d(c1, c2, 5, stride=1,padding=2),
            nn.ReLU(inplace=True),
        )
        self.lin1 = nn.Sequential(
            nn.Dropout(d2),
            nn.Linear(c2 *LEN ,L2),
            nn.ReLU(inplace=True),
            nn.Linear(L2 ,L3),
            nn.ReLU(inplace=True),
            nn.Linear(L3 ,out),
        )
        if args.hidden == "S":
            self.hidden = nn.Sigmoid()
        elif args.hidden =="R":
            self.hidden = nn.ReLU()
        elif args.hidden =="T":
            self.hidden = nn.Tanh()
        self.lin2 = nn.Sequential(
            nn.Linear(out, L3),
            nn.ReLU(inplace=True),
            nn.Linear(L3, L2),
            nn.ReLU(inplace=True),
            nn.Linear(L2, c2*LEN)
        )
        self.tra1 = nn.Sequential(
            nn.Dropout(d1),
            nn.ConvTranspose2d(c2,c1,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True),
        )
        self.up1 = nn.Upsample(scale_factor=p1, mode="bilinear")

        self.tra2 =  nn.Sequential(
            nn.Dropout(d1),
            nn.ConvTranspose2d(c1,inp,kernel_size=5,stride=1, padding=2),
        )
        if args.output == "S":
            self.output = nn.Sigmoid()
        elif args.output =="R":
            self.output = nn.ReLU()
        elif args.output =="T":
            self.output = nn.Tanh()

    def forward(self, x):
        co1 = self.conv1(x)
        po1 = self.MP1(co1)
        co2 = self.conv2(po1)
        co2 = co2.view(co2.size(0), -1)
        li1 = self.lin1(co2)
        encoded = self.hidden(li1)
        li2 = self.lin2(encoded)
        li2 = li2.view(li2.size(0),C2, LEN2 ,LEN2)
        t1 = self.tra1(li2)
        u1 = self.up1(t1)
        t2 = self.tra2(u1)
        decoded = self.output(t2)
        return encoded, decoded

def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    # p = F.softmax(p)
    # q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def train(model, args):
    rho = torch.FloatTensor([RHO for _ in range(10000)]).unsqueeze(0)
    print(rho)
    div = ""
    avg = ""
    log = ""
    diag = ""
    lognorm = ""
    if LOG:
        log = "_log"
    if LOGNORM:
        lognorm = "_lognorm"
    if DIVIDE:
        div = "_div"
    if AVG:
        avg = "_avg"
    if DIAG:
        diag = "_diag"
    tre = "_t"+str(args.treshold)
    autoencoder = MODEL_D + args.model
    train_set = pickle.load( open( SET_D+args.chrom+tre+log+lognorm+div+diag+avg+".p", "rb" ) )
    test_set = pickle.load( open(SET_D+args.chrom+tre+log+lognorm+div+diag+avg+"_test.p", "rb" ) )
    print(len(train_set))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batchSize,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batchSize,
        shuffle=False)

    if cuda: model.to(device)
    optimizer = torch.optim.Adam( model.parameters(), lr=args.learningRate)
    if args.mse:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    for epoch in range(args.epochs):
        start = time. time()
        for b_index, (x) in enumerate(train_loader):
            x = x[0]
            x = Variable(x)
            if cuda: x= x.to(device)
            encoded, decoded = model(x)
            loss = criterion(decoded,x)
            if use_sparse:
                # sqrt_encoded = torch.sqrt(encoded)
                # avg_encoded = torch.div(sqrt_encoded, 165)
                # rho_hat = torch.sum(avg_encoded, dim=0, keepdim=True)
                # rho_hat_old = torch.div(torch.sum(encoded,
                                                  # dim=0,keepdim=True),165)
                # sparsity_penalty = args.beta * torch.abs(torch.div(torch.sum(rho_hat)-
                    # torch.sum(rho),10000))
                # sparsity_penalty = 0.1 * torch.div(torch.abs(torch.sum(rho)
                                        # -torch.sum(rho_hat)),40000)
                # if sparsity_penalty < (0.01 *args.beta):
                    # sparsity_penalty = torch.tensor(0)
                rho_hat = torch.div(torch.sum(encoded, dim=0,
                                              keepdim=True),165)
                sparsity_penalty = args.beta * kl_divergence(rho, rho_hat)
                train_loss = loss + sparsity_penalty
            else:
                train_loss = loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            for b_index, (x) in enumerate(test_loader):
                x = x[0]
                x = Variable(x)
                if cuda: x= x.to(device)
                encoded, decoded = model(x)
                test_loss = criterion(decoded,x)
        if epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(),autoencoder  )
        end = time.time()
        if use_sparse:
            nonZeros = torch.nonzero(rho_hat).size(0)
            totalSum = torch.sum(rho_hat).data
            print("Epoch: [%3d], Loss: %.4f Val: %.4f SparsityLoss:%.4f NormalLoss: %.4f TotalSum: %.2f NonZeros: %.1f Time:%.2f" %(
                epoch +1,train_loss.data,test_loss.data,sparsity_penalty.data,loss.data,totalSum,nonZeros, end-start))
        else:
            print("Epoch: [%3d], Loss: %.4f Val: %.4f Time:%.2f" %(epoch +
                                                                   1,train_loss.data,test_loss.data, end-start))

        sys.stdout.flush()
    torch.save(model.state_dict(),autoencoder  )

