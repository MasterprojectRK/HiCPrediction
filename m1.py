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

torch.set_printoptions(precision=6)
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
ARGS = None
CUT_W =200
class SparseAutoencoder(nn.Module):
    def __init__(self,args ):
        super(SparseAutoencoder, self).__init__()
        self.args = args
        self.args.requires_grad = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.firstLayer, args.convSize, stride=1,
                      padding=args.padding)
            ,nn.ReLU(inplace=True)
        )
        self.MP1 = nn.MaxPool2d(args.pool1)
        self.conv2 = nn.Sequential(
            nn.Dropout(args.dropout1),
            nn.Conv2d(args.firstLayer, args.secondLayer, args.convSize, stride=1,
                      padding=args.padding),
            nn.ReLU(inplace=True),
        )
        self.MP2 = nn.MaxPool2d(args.pool1)
        self.conv3 = nn.Sequential(
            nn.Dropout(args.dropout1),
            nn.Conv2d(args.secondLayer, args.thirdLayer, args.convSize, stride=1,
                      padding=args.padding),
            nn.ReLU(inplace=True),
        )
        if args.lastLayer =="third":
                linInput = args.thirdLayer * args.dimAfterP2**2
        elif args.lastLayer =="second":
                linInput = args.secondLayer * args.dimAfterP1**2
        elif args.lastLayer =="thirdNoPool":
                linInput = args.thirdLayer * args.cutLength**2
        elif args.lastLayer =="secondNoPool":
                linInput = args.secondLayer * args.cutLength**2
        elif args.lastLayer =="first" :
                linInput = args.firstLayer * args.dimAfterP1**2
        elif args.lastLayer =="firstNoPool" :
                linInput = args.firstLayer * args.cutLength**2
        self.lin1 = nn.Sequential(
            nn.Dropout(args.dropout2),
            nn.Linear(linInput ,args.outputSize),
        )
    
        if args.hidden == "S":
            self.hidden = nn.Sigmoid()
        elif args.hidden =="R":
            self.hidden = nn.ReLU()
        elif args.hidden =="T":
            self.hidden = nn.Tanh()
        elif args.hidden =="TR":
            self.hidden = nn.Sequential(
                nn.Tanh(),
                nn.ReLU()
            )
        if args.lastLayer =="third":
                linInput = args.thirdLayer * args.dimAfterP2**2
        if args.lastLayer =="thirdNoPool":
                linInput =args.thirdLayer * args.cutLength**2
        elif args.lastLayer =="second":
                linInput = args.secondLayer * args.dimAfterP1**2
        elif args.lastLayer =="secondNoPool":
                linInput = args.secondLayer * args.cutLength**2
        elif args.lastLayer =="first" :
                linInput = args.firstLayer * args.dimAfterP1**2
        elif args.lastLayer =="firstNoPool" :
                linInput = args.firstLayer * args.cutLength**2
        
        self.lin2 = nn.Sequential(
            nn.Linear(args.outputSize, linInput),
            nn.ReLU(inplace=True),
        )
    
        self.tra1 = nn.Sequential(
            nn.Dropout(args.dropout1),
            nn.ConvTranspose2d(args.thirdLayer, args.secondLayer,kernel_size=args.convSize,
                               stride=1,padding=args.padding),
            nn.ReLU(True),
        )
        self.up1 = nn.Upsample(scale_factor=args.pool1, mode="bilinear")
        self.tra2 =  nn.Sequential(
            nn.Dropout(args.dropout1),
            nn.ConvTranspose2d(args.secondLayer, args.firstLayer,kernel_size=args.convSize,
                               stride=1,padding=args.padding),
        )
        self.up2 = nn.Upsample(scale_factor=args.pool1, mode="bilinear")
        self.tra3 =  nn.Sequential(
            nn.Dropout(args.dropout1),
            nn.ConvTranspose2d(args.firstLayer,1,kernel_size=args.convSize,stride=1,
                               padding=args.padding),
        )
        if args.output == "S":
            self.output = nn.Sigmoid()
        elif args.output =="R":
            self.output = nn.ReLU()
        elif args.output =="T":
            self.output = nn.Tanh()
        elif args.output =="TR":
            self.output = nn.Sequential(
                nn.Tanh(),
                nn.ReLU()
            )

    def forward(self, x):
        if self.args.lastLayer =="first":
            co1 = self.conv1(x)
            po1 = self.MP1(co1)
            po1_m = po1.view(po1.size(0), -1)
            li1 = self.lin1(po1_m)
            encoded = self.hidden(li1)
            li2 = self.lin2(encoded)
            li2 = li2.view(po1.size())
            u2 = self.up2(li2)
            t3 = self.tra3(u2)
        elif self.args.lastLayer =="firstNoPool":
            co1 = self.conv1(x)
            co1_m = co1.view(co1.size(0), -1)
            li1 = self.lin1(co1_m)
            encoded = self.hidden(li1)
            li2 = self.lin2(encoded)
            li2 = li2.view(co1.size())
            t3 = self.tra3(li2)
        elif self.args.lastLayer =="secondNoPool":
            co1 = self.conv1(x)
            co2 = self.conv2(co1)
            co2_m = co2.view(co2.size(0), -1)
            li1 = self.lin1(co2_m)
            encoded = self.hidden(li1)
            li2 = self.lin2(encoded)
            li2 = li2.view(co2.size())
            t2 = self.tra2(li2)
            t3 = self.tra3(t2)
        elif self.args.lastLayer =="second":
            co1 = self.conv1(x)
            po1 = self.MP1(co1)
            co2 = self.conv2(po1)
            co2_m = co2.view(co2.size(0), -1)
            li1 = self.lin1(co2_m)
            encoded = self.hidden(li1)
            li2 = self.lin2(encoded)
            li2 = li2.view(co2.size())
            t2 = self.tra2(li2)
            u2 = self.up2(t2)
            t3 = self.tra3(u2)
        elif self.args.lastLayer =="thirdNoPool":
            co1 = self.conv1(x)
            co2 = self.conv2(co1)
            co3 = self.conv3(co2)
            co3_m = co3.view(co3.size(0), -1)
            li1 = self.lin1(co3_m)
            encoded = self.hidden(li1)
            li2 = self.lin2(encoded)
            li2 = li2.view(co3.size())
            t1 = self.tra1(li2)
            t2 = self.tra2(t1)
            t3 = self.tra3(t2)
        elif self.args.lastLayer =="third":
            co1 = self.conv1(x)
            po1 = self.MP1(co1)
            co2 = self.conv2(po1)
            po2 = self.MP2(co2)
            co3 = self.conv3(po2)
            co3_m = co3.view(co3.size(0), -1)
            li1 = self.lin1(co3_m)
            encoded = self.hidden(li1)
            li2 = self.lin2(encoded)
            li2 = li2.view(co3.size())
            t1 = self.tra1(li2)
            u1 = self.up1(t1)
            t2 = self.tra2(u1)
            u2 = self.up2(t2)
            t3 = self.tra3(u2)
        
        decoded = self.output(t3)
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
    tre = "_t"+str(args.treshold)
    autoencoder = MODEL_D + args.model
    print(args.prepData)
    train_set = pickle.load( open( SET_D+args.chrom+tre+"_"+args.prepData+".p", "rb" ) )
    test_set = pickle.load( open(SET_D+args.chrom+tre+"_"+args.prepData+"_test.p", "rb" ) )
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
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "BCE":
        criterion = nn.BCELoss()
    
    valCriterion = nn.L1Loss()
    for epoch in range(args.epochs):
        start = time. time()
        for b_index, (x) in enumerate(train_loader):
            x = x[0]
            x = Variable(x)
            # print(x[0][0][0].data)
            if cuda: x= x.to(device)
            encoded, decoded = model(x)
            # print(decoded.shape)
            # print(encoded.reshape(x.shape))
            loss = criterion(decoded,x)
            l1Loss = valCriterion(decoded,x)
            # print(decoded[0][0][0].data)
            # print("decoded")
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
                                              keepdim=True),150)
                sparsity_penalty = args.beta * kl_divergence(rho, rho_hat)
                train_loss = loss + sparsity_penalty
            else:
                train_loss = loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # if (args.saveModel and epoch % 10 == 0) or (epoch == args.epochs -1):
        if True:
            for b_index, (x) in enumerate(test_loader):
                x = x[0]
                x = Variable(x)
                if cuda: x= x.to(device)
                encoded, decoded = model(x)
                test_loss = valCriterion(decoded,x)
        if args.saveModel and epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(),autoencoder  )
        end = time.time()
        if  use_sparse:
            nonZeros = torch.nonzero(rho_hat).size(0)
            totalSum = torch.sum(rho_hat).data
            print("Epoch: [%3d], Loss: %.4f Val: %.4f SparsityLoss:%.4f NormalLoss: %.4f TotalSum: %.2f NonZeros: %.1f Time:%.2f" %(
                epoch
                +1,train_loss.data,test_loss.data,sparsity_penalty.data,loss.data,totalSum,nonZeros,
                end-start), end="")
       
        print("Epoch: [%3d], Loss: %.4f l1Loss: %.4f Val: %.4f Time:%.2f" %(epoch +
            1,train_loss.data,l1Loss.data,test_loss.data, end-start),end="")
        print('\r', end='')                     # use '\r' to go back

        sys.stdout.flush()
    if args.saveModel:
        torch.save(model.state_dict(),autoencoder  )
    return float(test_loss.data)
