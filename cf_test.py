#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cf_loader

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "../dataset/450_train.csv"
test_file = "../dataset/test.csv"
output_file = "./test_output1.csv"

# Model Hyperparameters
user_dim = song_dim = 8

# Training Parameters
learning_rate = 0.0005
batch_size = 100
num_epochs = 100
evaluate_every = 3

default = 0.0

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, song_cnt, user_train, song_train, target_train, idx_test, user_test, song_test, target_test = cf_loader.load_data(train_file, test_file, 1)

print("train/test/: {:d}/{:d}".format(len(user_train), len(user_test)))
print("==================================================================================")

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.user_weight = nn.Embedding(user_cnt, user_dim).type(ftype)
        #self.user_weight.weight.data.copy_(torch.from_numpy(np.load("weight/cf_user_211.npy")))
        self.song_weight = nn.Embedding(song_cnt, song_dim).type(ftype)
        #self.song_weight.weight.data.copy_(torch.from_numpy(np.load("weight/cf_song_211.npy")))
        self.relu = nn.ReLU()

    def forward(self, user, song):
        user = self.user_weight(user)
        song = self.song_weight(song)
        x = torch.sum(torch.mul(user, song), dim=1)
        x = self.relu(x)
        return x

###############################################################################################
def parameters():

    params = []
    for model in [cf_model]:
        params += list(model.parameters())

    return params

def run(idx, msno, songid, step):

    optimizer.zero_grad()

    idx = Variable(torch.from_numpy(np.asarray(idx))).type(ftype)
    msno = Variable(torch.from_numpy(np.asarray(msno))).type(ltype)
    songid = Variable(torch.from_numpy(np.asarray(songid))).type(ltype)
    #target = Variable(torch.from_numpy(np.asarray(target))).type(ftype)

    # Linears 
    cf_out = cf_model(msno, songid)

    if step == 3:
        cf_out = torch.stack([idx, cf_out], 1)
        return cf_out

    # MSE
    loss = loss_model(cf_out, idx)

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()

def print_score(batches, step):

    total_out = []
    for j, batch in enumerate(batches):
        idx_batch, user_batch, song_batch = zip(*batch)
        batch_out = run(idx_batch, user_batch, song_batch, step=step)
        total_out.append(batch_out)
    total_out = torch.cat(total_out, 0)

    test_wf = open(output_file, 'w')
    i = 0
    for idx, prob in total_out.data.cpu().numpy():
        while(1):
            if prob > 1: prob = 1.0
            if i == int(idx):
                test_wf.write(str(int(idx))+","+str(prob)+'\n')
                i += 1
                break
            else:
                test_wf.write(str(i)+","+str(default)+'\n')
                i += 1 

###############################################################################################
cf_model = Linear().cuda()
loss_model = nn.MSELoss()
optimizer = torch.optim.Adam(parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.9)
#optimizer = torch.optim.ASGD(parameters(), lr=learning_rate, alpha=0.9)

for i in xrange(num_epochs):
    # Training
    train_batches = cf_loader.train_batch_iter(list(zip(user_train, song_train, target_train)), batch_size)
    loss = 0.
    for j, train_batch in enumerate(train_batches):
        msno_batch, song_batch, target_batch = zip(*train_batch)
        loss += run(target_batch, msno_batch, song_batch, step=1)
        if (j+1) % 30000 == 0:
            print("Training at epoch #{}: ".format(j+1)), "batch_mse :", loss/j, datetime.datetime.now()

# Testing
test_batches = cf_loader.validation_batch_iter(list(zip(idx_test, user_test, song_test)), batch_size)
print_score(test_batches, step=3)
