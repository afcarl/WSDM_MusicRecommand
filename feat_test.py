#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import feat_loader as feat_loader

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "../dataset/450_train.csv"
test_file = "../dataset/test.csv"
song_file = "../dataset/songs.csv"
output_file = "./feat_output1.csv"

# Model Hyperparameters
user_dim = song_dim = 16
sst_dim = ssn_dim = st_dim = 1
genre_dim = artist_dim = 8

# Training Parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 30
evaluate_every = 3

default = 0.0

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, song_cnt, genre_cnt, artist_cnt, sst_cnt, ssn_cnt, st_cnt, user_train, song_train, sst_train, ssn_train, st_train, target_train, song_tune, genre_tune, artist_tune, idx_test, user_test, genre_test, artist_test, sst_test, ssn_test, st_test = feat_loader.load_data(train_file, test_file, song_file)

print("train/tune/test/user/song/genre/artist/sst/ssn/st: {:d}/{:d}/{:d}/{:d}/{:d}/{:d}/{:d}/{:d}/{:d}/{:d}".format(len(user_train), len(song_tune), len(user_test), user_cnt, song_cnt, genre_cnt, artist_cnt, sst_cnt, ssn_cnt, st_cnt))
print("==================================================================================")

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.user_weight = nn.Embedding(user_cnt, user_dim).type(ftype)
        self.song_weight = nn.Embedding(song_cnt, song_dim).type(ftype)
        self.sst_weight = nn.Embedding(sst_cnt, sst_dim).type(ftype)
        self.ssn_weight = nn.Embedding(ssn_cnt, ssn_dim).type(ftype)
        self.st_weight = nn.Embedding(st_cnt, st_dim).type(ftype)
        self.genre_weight = nn.Embedding(genre_cnt, genre_dim).type(ftype)
        self.artist_weight = nn.Embedding(artist_cnt, artist_dim).type(ftype)

        # 19 > 8 > 1
        self.linear1 = nn.Linear(user_dim+song_dim+sst_dim+ssn_dim+st_dim, 8)
        self.linear2 = nn.Linear(8,2)
        self.linear3 = nn.Linear(genre_dim+artist_dim, song_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def train(self, user, song, sst, ssn, st):
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

    def tune(self, song, genre, artist):
        song = self.song_weight(song)
        genre = self.genre_weight(genre)
        artist = self.artist_weight(artist)

        input_ = torch.cat([genre, artist], 1)
        output = self.tanh(self.linear3(input_))

        return song, output

    def test(self, user, genre, artist, sst, ssn, st):
        user = self.user_weight(user)
        genre = self.genre_weight(genre)
        artist = self.artist_weight(artist)
        song = self.tanh(self.linear3(torch.cat([genre, artist], 1)))
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
    sst = Variable(torch.from_numpy(np.asarray(sst))).type(ltype)
    ssn = Variable(torch.from_numpy(np.asarray(ssn))).type(ltype)
    st = Variable(torch.from_numpy(np.asarray(st))).type(ltype)

    if step < 3: 
        songid = Variable(torch.from_numpy(np.asarray(songid))).type(ltype)
        lin_out = lin_model.train(msno, songid, sst, ssn, st)[:,1]
    else:
        genre = Variable(torch.from_numpy(np.asarray(songid[0]))).type(ltype)
        artist = Variable(torch.from_numpy(np.asarray(songid[1]))).type(ltype)
        lin_out = lin_model.test(msno, genre, artist, sst, ssn, st)[:,1]
        lin_out = torch.stack([idx, lin_out], 1)
        return lin_out.data.cpu().numpy()

    # MSE
    loss = loss_model(lin_out, idx)

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()

def tune(song, genre, artist):

    optimizer.zero_grad()

    song = Variable(torch.from_numpy(np.asarray(song))).type(ltype)
    genre = Variable(torch.from_numpy(np.asarray(genre))).type(ltype)
    artist = Variable(torch.from_numpy(np.asarray(artist))).type(ltype)

    song, lin_out = lin_model.tune(song, genre, artist)
    loss = F.mse_loss(song, lin_out)

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()

def print_score(batches, step):

    total_out = []
    for j, batch in enumerate(batches):
        idx_batch, user_batch, genre_batch, artist_batch, sst_batch, ssn_batch, st_batch = zip(*batch)
        batch_out = run(idx_batch, user_batch, (genre_batch, artist_batch), sst_batch, ssn_batch, st_batch, step=step)
        total_out.append(batch_out)
    total_out = np.concatenate(total_out, 0)

    test_wf = open(output_file, 'w')
    i = 0
    unk = 0
    for idx, prob in total_out:
        if prob > 1: prob = 1.0
        if prob < 0:
            prob = 0.0
            unk += 1
        while(1):
            if i == int(idx):
                test_wf.write(str(int(idx))+","+str(prob)+'\n')
                i += 1
                break
            else:
                test_wf.write(str(i)+","+str(default)+'\n')
                i += 1 
    print "unknown cnt :", unk

###############################################################################################
lin_model = Linear().cuda()
loss_model = nn.MSELoss()
optimizer = torch.optim.Adam(parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.9)
#optimizer = torch.optim.ASGD(parameters(), lr=learning_rate, alpha=0.9)

user_cnt, song_cnt, genre_cnt, artist_cnt, sst_cnt, ssn_cnt, st_cnt, train_user, train_song, train_sst, train_ssn, train_st, train_target, tune_song, tune_genre, tune_artist, test_idx, test_user, test_genre, test_artist, test_sst, test_ssn, test_st = feat_loader.load_data(train_file, test_file, song_file)

# User Song Training
for i in xrange(num_epochs):
    train_batches = feat_loader.train_batch_iter(list(zip(user_train, song_train, sst_train, ssn_train, st_train, target_train)), batch_size)
    loss = 0.
    for j, train_batch in enumerate(train_batches):
        msno_batch, song_batch, sst_batch, ssn_batch, st_batch, target_batch = zip(*train_batch)
        loss += run(target_batch, msno_batch, song_batch, sst_batch, ssn_batch, st_batch, step=1)
        if (j+1) % 10000 == 0:
            print("Training at epoch #{}: ".format(j+1)), "batch_mse :", loss/j, datetime.datetime.now()

print("==================================================================================")
print("Tuning...")
# Song Feature Training
for i in xrange(num_epochs):
    train_batches = feat_loader.train_batch_iter(list(zip(song_tune, genre_tune, artist_tune)), batch_size)
    loss = 0.
    for j, train_batch in enumerate(train_batches):
        song_batch, genre__batch, artist_batch = zip(*train_batch)
        loss += tune(song_batch, genre__batch, artist_batch)
    print("Tuning at epoch #{}: ".format(j+1)), "batch_mse :", loss/j, datetime.datetime.now()

# Testing
print("==================================================================================")
print("Testing...")
test_batches = feat_loader.validation_batch_iter(list(zip(idx_test, user_test, genre_test, artist_test, sst_test, ssn_test, st_test)), batch_size)
print_score(test_batches, step=3)
