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
BIN = 10000
CHR = "4"
DATA_D = "Data2e/"
CHROM_D = DATA_D + "Chroms/"
SET_D = DATA_D + "Sets/"
PRED_D = DATA_D + "Predictions/"
MODEL_D  = DATA_D + "Models/"
CHUNK_D = DATA_D + "Chunks/"
TEST_D =  DATA_D + "Test/"
IMAGE_D = DATA_D +  "Images/"
ORIG_D = DATA_D +  "Orig/"
#BETA = 0.0001 
BETA = 1 
RHO = 0.3
use_sparse = True

LENGTH = 200
DIAG = False
CUT_W =200
DIVIDE =  False
AVG = False
SPARSE= False
LOG = True
LOGNORM = False
MAX = 4846
D1 = 0.1
D2 = 0.5
IN = 1
C1 = 2
C2 = 2
C3 = 2
P1 = 2
L1 = 40000 
OUT = 40000
class SparseAutoencoder(nn.Module):
    def __init__(self,c1=C1, c2=C2, c3=C3, p1=P1, l1=L1, out=OUT,
                 d1=D1,d2=D2, inp=IN ):
        super(SparseAutoencoder, self).__init__()
        #self.conv1 = nn.Sequential(
        self.conv1 = nn.Conv2d(inp, c1, 5, stride=1, padding=2)
        #    ,nn.ReLU(inplace=True)
        #)
        self.MP1 = nn.MaxPool2d(p1)
        # self.conv2 = nn.Sequential(
            # nn.Dropout(d1),
            # nn.Conv2d(c1, c2, 5, stride=1,padding=2),
            # nn.ReLU(inplace=True),
        # )
        # self.conv3 = nn.Sequential(
            # nn.Dropout(d1),
            # nn.Conv2d(c2, c3, 5, stride=1,padding=2),
            # nn.ReLU(inplace=True),
        # )
        self.lin1 = nn.Sequential(
            nn.Dropout(d2),
            nn.Linear(c1 * 100 *100,OUT),
            nn.Sigmoid()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(out, c1 * 100 * 100)
        )
        # self.tra1 = nn.Sequential(
            # nn.Dropout(d1),
            # nn.ConvTranspose2d(c3, c2,kernel_size=5,stride=1, padding=2),
            # nn.ReLU(True),
        # )
        # self.tra2 = nn.Sequential(
            # nn.Dropout(d1),
            # nn.ConvTranspose2d(c2,c1,kernel_size=5,stride=1, padding=2),
            # nn.ReLU(True),
        # )
        self.up1 = nn.Upsample(scale_factor=p1, mode="bilinear")

        self.tra3 =  nn.Sequential(
            nn.Dropout(d1),
            nn.ConvTranspose2d(c1,IN,kernel_size=5,stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        #log.debug(x.shape)
        #print(self.W.shape)
        #print(self.W[0][0])
        co1 = self.conv1(x)
        ("c1",co1.shape)
        po1 = self.MP1(co1)
        #log.debug("po1",po1.shape)
        #c2 = self.conv2(p1)
        #log.debug("co2",co2.shape)
        #co3 = self.conv3(co2)
        co3 = po1
        #log.debug("co3",co3.shape)
        co3 = co3.view(co3.size(0), -1)
        li1 = self.lin1(co3)
        encoded = li1
        #log.debug("enc",encoded.shape)
        li2 = self.lin2(encoded)
        li2 = li2.view(li2.size(0),C1, 100 ,100)
        #log.debug("l2",l2.shape)
        #t1 = self.tra1(l2)
        #log.debug("t1",t1.shape)
        #t2 = self.tra2(t1)
        #log.debug("t2",t2.shape)
        u1 = self.up1(li2)
        #log.debug("u1",u1.shape)
        t3 = self.tra3(u1)
        #log.debug("t3",t3.shape)
        decoded = t3
        #log.debug("dec",decoded.shape)
        return encoded, decoded




def train(model, args):
    rho = torch.FloatTensor([RHO for _ in range(40000)]).unsqueeze(0)
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
    autoencoder = MODEL_D + args.model
    train_set = pickle.load( open( SET_D+args.chrom+log+lognorm+div+diag+avg+".p", "rb" ) )
    test_set = pickle.load( open( SET_D+args.chrom+log+lognorm+div+diag+avg+"_test.p", "rb" ) )
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
                sqrt_encoded = torch.sqrt(encoded)
                avg_encoded = torch.div(sqrt_encoded, 115)
                rho_hat = torch.sum(avg_encoded, dim=0, keepdim=True)
                rho_hat_old = torch.div(torch.sum(encoded, dim=0,keepdim=True),115)
                print(rho_hat)
                print(torch.sum(rho))
                print(torch.sum(rho_hat))
                print(torch.sum(rho_hat_old))
                # sparsity_penalty = BETA * F.kl_div(rho, rho_hat)
                sparsity_penalty = torch.abs(torch.div(torch.sum(rho_hat)-
                    torch.sum(rho),40000))
                # sparsity_penalty = 0.1 * torch.div(torch.abs(torch.sum(rho)
                                        # -torch.sum(rho_hat)),40000)
                print(sparsity_penalty)
                if sparsity_penalty < 0.005:
                    sparsity_penalty = 0
                print(loss)
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
        end = time. time()
        print("Epoch: [%3d], Loss: %.4f Val: %.4f Time:%.4f" %(epoch + 1, train_loss.data,
                                                        test_loss.data,
                                                               end-start))
        sys.stdout.flush()
    torch.save(model.state_dict(),autoencoder  )

