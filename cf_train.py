#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
import cf_loader

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "../dataset/450_train.csv"
test_file = "../dataset/test.csv"

# Model Hyperparameters
# 1: 32dim 100batch 60epoch 0.001lr Adam 450_train.csv | 4dim 8dim pool perform
# 2: 64dim 100batch 60epoch 0.001lr Adam
# 3: 16dim 100batch 60epoch 0.0005lr Adam
# 4: 8dim 100batch 60epoch 0.001lr Adam
# 5: 8dim 100batch 60epoch 0.005lr Adam
# 6: 8dim 64batch 60epoch 0.001lr Adam
# 7: 8dim 32batch 60epoch 0.001lr Adam
# 8: 8dim 100batch 60epoch 0.001lr Adam train.csv
# 3: 16dim 100batch 60epoch 0.001lr Adam 450_train.csv
user_dim = song_dim = 16

# Training Parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 60
evaluate_every = 3

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, song_cnt, user_train, song_train, target_train, idx_test, user_test, song_test, target_test = cf_loader.load_data(train_file, test_file, 0)

print("train/test/: {:d}/{:d}".format(len(user_train), len(user_test)))
print("==================================================================================")

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.user_weight = nn.Embedding(user_cnt, user_dim).type(ftype)
        #self.user_weight.weight.data.copy_(torch.from_numpy(np.load("weight/cf_user.npy")))
        self.song_weight = nn.Embedding(song_cnt, song_dim).type(ftype)
        #self.song_weight.weight.data.copy_(torch.from_numpy(np.load("weight/cf_song.npy")))
        self.relu = nn.ReLU()
        #self.lsoftmax = nn.LogSoftmax()

    def forward(self, user, song):
        user = self.user_weight(user)
        song = self.song_weight(song)
        x = torch.sum(torch.mul(user, song), dim=1)
        x = self.relu(x)
        #x = self.lsoftmax(x)
        return x

###############################################################################################
def parameters():

    params = []
    for model in [cf_model]:
        params += list(model.parameters())

    return params

def run(msno, songid, target, step):

    optimizer.zero_grad()

    msno = Variable(torch.from_numpy(np.asarray(msno))).type(ltype)
    songid = Variable(torch.from_numpy(np.asarray(songid))).type(ltype)
    target = Variable(torch.from_numpy(np.asarray(target))).type(ftype)

    # Linears 
    cf_out = cf_model(msno, songid)

    # MSE
    loss = loss_model(cf_out, target)

    if step > 1:
        # AUC
        fpr, tpr, _ = metrics.roc_curve(target.data.cpu().numpy(), cf_out.data.cpu().numpy())
        auc = metrics.auc(fpr, tpr)
        return auc, loss.data.cpu().numpy()

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()

def print_score(batches, step):
    total_loss = 0.0
    total_auc = 0.0

    for j, batch in enumerate(batches):
        user_batch, song_batch, target_batch = zip(*batch)
        batch_auc, batch_loss = run(user_batch, song_batch, target_batch, step=step)
        total_auc += batch_auc
        total_loss += batch_loss

    total_auc = total_auc/j
    total_loss = total_loss/j
    print("auc : {}, mse : {}: ".format(total_auc, total_loss)), datetime.datetime.now()

    if step == 3:
        np.save("weight/cf_user4.npy", cf_model.user_weight.weight.data.cpu().numpy())
        np.save("weight/cf_song4.npy", cf_model.song_weight.weight.data.cpu().numpy())

###############################################################################################
cf_model = Linear().cuda()
#loss_model = nn.CrossEntropyLoss()
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
        loss += run(msno_batch, song_batch, target_batch, step=1)

    # Validation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{}".format(i+1)), datetime.datetime.now()
        valid_batches = cf_loader.validation_batch_iter(list(zip(user_test, song_test, target_test)), batch_size)
        print_score(valid_batches, step=2)


# Testing
print("==================================================================================")
print("Training End..")
print("Test: ")
test_batches = cf_loader.validation_batch_iter(list(zip(user_test, song_test, target_test)), batch_size)
print_score(test_batches, step=3)
