#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cf_loader_extra as cf_loader

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "../dataset/train.csv"
test_file = "../dataset/test.csv"
output_file = "./test_output5.csv"

# Model Hyperparameters
user_dim = song_dim = 8
sst_dim = ssn_dim = st_dim = 1

# Training Parameters
learning_rate = 0.0005
batch_size = 30
num_epochs = 100
evaluate_every = 3

default = 0.0

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, song_cnt, sst_cnt, ssn_cnt, st_cnt, user_train, song_train, sst_train, ssn_train, st_train, target_train, idx_test, user_test, song_test, sst_test, ssn_test, st_test, target_test = cf_loader.load_data(train_file, test_file, 1)

print("train/test/: {:d}/{:d}".format(len(user_train), len(user_test)))
print("==================================================================================")

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.user_weight = nn.Embedding(user_cnt, user_dim).type(ftype)
        self.song_weight = nn.Embedding(song_cnt, song_dim).type(ftype)
        self.sst_weight = nn.Embedding(sst_cnt, sst_dim).type(ftype)
        self.ssn_weight = nn.Embedding(ssn_cnt, ssn_dim).type(ftype)
        self.st_weight = nn.Embedding(st_cnt, st_dim).type(ftype)
        self.tanh = nn.Tanh()
        # 19 > 8 > 1
        self.linear1 = nn.Linear(user_dim+song_dim+sst_dim+ssn_dim+st_dim, 8)
        self.linear2 = nn.Linear(8,2)
        self.softmax = nn.Softmax()

    def forward(self, user, song, sst, ssn, st):
        user = self.user_weight(user)
        song = self.song_weight(song)
        sst = self.sst_weight(sst)
        ssn = self.ssn_weight(ssn)
        st = self.st_weight(st)

        input_ = torch.cat([user,song,sst,ssn,st], 1)
        output = self.tanh(self.linear1(input_))        
        output = self.tanh(self.linear2(output))        
        output = self.softmax(output)

        return output

###############################################################################################
def parameters():

    params = []
    for model in [lin_model]:
        params += list(model.parameters())

    return params

def run(idx, msno, songid, sst, ssn, st, step):

    optimizer.zero_grad()

    idx = Variable(torch.from_numpy(np.asarray(idx))).type(ftype)
    msno = Variable(torch.from_numpy(np.asarray(msno))).type(ltype)
    songid = Variable(torch.from_numpy(np.asarray(songid))).type(ltype)
    sst = Variable(torch.from_numpy(np.asarray(sst))).type(ltype)
    ssn = Variable(torch.from_numpy(np.asarray(ssn))).type(ltype)
    st = Variable(torch.from_numpy(np.asarray(st))).type(ltype)

    # Linears 
    lin_out = lin_model(msno, songid, sst, ssn, st)
    lin_out = lin_out[:,1]

    if step == 3:
        lin_out = torch.stack([idx, lin_out], 1)
        return lin_out

    # MSE
    loss = loss_model(lin_out, idx)

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()

def print_score(batches, step):

    total_out = []
    for j, batch in enumerate(batches):
        idx_batch, user_batch, song_batch, sst_batch, ssn_batch, st_batch = zip(*batch)
        batch_out = run(idx_batch, user_batch, song_batch, sst_batch, ssn_batch, st_batch, step=step)
        total_out.append(batch_out)
    total_out = torch.cat(total_out, 0)

    test_wf = open(output_file, 'w')
    i = 0
    for idx, prob in total_out.data.cpu().numpy():
        if prob > 1: prob = 1.0
        if prob < 0: prob = 0.0
        while(1):
            if i == int(idx):
                test_wf.write(str(int(idx))+","+str(prob)+'\n')
                i += 1
                break
            else:
                test_wf.write(str(i)+","+str(default)+'\n')
                i += 1 

###############################################################################################
lin_model = Linear().cuda()
loss_model = nn.MSELoss()
optimizer = torch.optim.Adam(parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.9)
#optimizer = torch.optim.ASGD(parameters(), lr=learning_rate, alpha=0.9)

for i in xrange(num_epochs):
    # Training
    train_batches = cf_loader.train_batch_iter(list(zip(user_train, song_train, sst_train, ssn_train, st_train, target_train)), batch_size)
    loss = 0.
    for j, train_batch in enumerate(train_batches):
        msno_batch, song_batch, sst_batch, ssn_batch, st_batch, target_batch = zip(*train_batch)
        loss += run(target_batch, msno_batch, song_batch, sst_batch, ssn_batch, st_batch, step=1)
        if (j+1) % 30000 == 0:
            print("Training at epoch #{}: ".format(j+1)), "batch_mse :", loss/j, datetime.datetime.now()

# Testing
test_batches = cf_loader.validation_batch_iter(list(zip(idx_test, user_test, song_test, sst_test, ssn_test, st_test)), batch_size)
print_score(test_batches, step=3)
