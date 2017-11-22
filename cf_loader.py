import numpy as np
import random

def load_data(train, test, step):

    user_dict = {}
    song_dict = {}

    train_f = open(train, "r")
    lines = train_f.readlines()[1:]
    random.shuffle(lines)
    train_len = len(lines)

    train_user = []
    train_song = []
    train_target = []
    test_idx = []
    test_user = []
    test_song = []
    test_target = []

    train_thr = len(lines) * 0.8

    for i, line in enumerate(lines):

        tokens = line.strip().split(',')
        msno = tokens[0]
        song_id = tokens[1]
        sst = tokens[2]
        ssn = tokens[3]
        st = tokens[4]
        target = int(tokens[5])

        if user_dict.get(msno) is None:
            user_dict[msno] = len(user_dict)
        if song_dict.get(song_id) is None:
            song_dict[song_id] = len(song_dict)

        if i < train_thr:
            train_user.append(user_dict.get(msno))
            train_song.append(song_dict.get(song_id))
            train_target.append(target)
        else:
            test_user.append(user_dict.get(msno))
            test_song.append(song_dict.get(song_id))
            test_target.append(target)

    if step == 1:
        test_f = open(test, "r")
        lines = test_f.readlines()[1:]
        test_len = len(lines)

        unk = 0
        for i, line in enumerate(lines):

            tokens = line.strip().split(',')
            idx = int(tokens[0])
            msno = tokens[1]
            song_id = tokens[2]
            sst = tokens[3]
            ssn = tokens[4]
            st = tokens[5]

            #print msno, song_id
            msno = user_dict.get(msno)
            song_id = song_dict.get(song_id)

            #print msno, song_id
            if (msno is None) or (song_id is None):
                unk += 1
                #print "!UNKNOWN WARNING :", msno, song_id, idx
                continue
            else:
                test_idx.append(idx)
                test_user.append(msno)
                test_song.append(song_id)
                #test_sst.append(sst)
                #test_ssn.append(ssn)
                #test_st.append(st)
        print "unknown test :", unk

    return len(user_dict), len(song_dict), train_user, train_song, train_target, test_idx, test_user, test_song, test_target

def train_batch_iter(data, batch_size):
    random.shuffle(data)
    data_size = len(data)
    num_batches = int(len(data)/batch_size) + 1 
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]

def validation_batch_iter(data, batch_size):
    data = np.array(data)
    data_size = len(data)
    num_batches = int(len(data)/batch_size) + 1 
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
