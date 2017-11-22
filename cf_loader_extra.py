import numpy as np
import random

user_dict = {'unk':0}
song_dict = {'unk':0}

song_f = open("../dataset/songs.csv", 'r')
lines = song_f.readlines()[1:]
for i, line in enumerate(lines):
    tokens = line.strip().split(',')
    song_id = tokens[0]
    if song_dict.get(song_id) is None:
        song_dict[song_id] = len(song_dict)

mem_f = open("../dataset/members.csv", 'r')
lines = mem_f.readlines()[1:]
for i, line in enumerate(lines):
    tokens = line.strip().split(',')
    msno = tokens[0]
    if user_dict.get(msno) is None:
        user_dict[msno] = len(user_dict)

def load_data(train, test, step):

    sst_dict = {'unk':0}
    ssn_dict = {'unk':0}
    st_dict = {'unk':0}

    train_f = open(train, "r")
    lines = train_f.readlines()[1:]
    random.shuffle(lines)
    train_len = len(lines)

    train_user = []
    train_song = []
    train_sst = []
    train_ssn = []
    train_st = []
    train_target = []
    test_idx = []
    test_user = []
    test_song = []
    test_sst = []
    test_ssn = []
    test_st = []
    test_target = []

    train_thr = len(lines) * 1.0

    for i, line in enumerate(lines):

        tokens = line.strip().split(',')
        msno = tokens[0]
        song_id = tokens[1]
        sst = tokens[2]
        ssn = tokens[3]
        st = tokens[4]
        target = int(tokens[5])

        if song_dict.get(song_id) is None:
            song_dict[song_id] = len(song_dict)
        if user_dict.get(msno) is None:
            user_dict[msno] = len(user_dict)
        if sst_dict.get(sst) is None:
            sst_dict[sst] = len(sst_dict)
        if ssn_dict.get(ssn) is None:
            ssn_dict[ssn] = len(ssn_dict)
        if st_dict.get(st) is None:
            st_dict[st] = len(st_dict)

        if i < train_thr:
            train_user.append(user_dict.get(msno))
            train_song.append(song_dict.get(song_id))
            train_sst.append(sst_dict.get(sst))
            train_ssn.append(ssn_dict.get(ssn))
            train_st.append(st_dict.get(st))
            train_target.append(target)
        else:
            test_user.append(user_dict.get(msno))
            test_song.append(song_dict.get(song_id))
            test_sst.append(sst_dict.get(sst))
            test_ssn.append(ssn_dict.get(ssn))
            test_st.append(st_dict.get(st))
            test_target.append(target)

    if step == 1:
        test_f = open(test, "r")
        lines = test_f.readlines()[1:]
        test_len = len(lines)

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
            sst = sst_dict.get(sst)
            ssn = ssn_dict.get(ssn)
            st = st_dict.get(st)

            #print msno, song_id
            if (msno is None):
                msno = user_dict.get('unk')
            if (song_id is None):
                song_id = song_dict.get('unk')
            if (sst is None):
                sst = sst_dict.get('unk')
            if (ssn is None):
                ssn = ssn_dict.get('unk')
            if (st is None):
                st = st_dict.get('unk')
                
            test_idx.append(idx)
            test_user.append(msno)
            test_song.append(song_id)
            test_sst.append(sst)
            test_ssn.append(ssn)
            test_st.append(st)

    return len(user_dict), len(song_dict), len(sst_dict), len(ssn_dict), len(st_dict), train_user, train_song, train_sst, train_ssn, train_st, train_target, test_idx, test_user, test_song, test_sst, test_ssn, test_st, test_target

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
