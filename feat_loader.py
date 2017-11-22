import pandas as pd
import numpy as np
import gc
import random

user_dict = {'unk':0}
song_dict = {'unk':0}
genre_dict = {'unk':0}
artist_dict = {'unk':0}
composer_dict = {'unk':0}
lyricist_dict = {'unk':0}
language_dict = {'unk':0}
sst_dict = {'unk':0}
ssn_dict = {'unk':0}
st_dict = {'unk':0}

song_f = open("../dataset/songs.csv", 'r')
lines = song_f.readlines()[1:]
for i, line in enumerate(lines):
    tokens = line.strip().split(',')
    song_id = tokens[0]
    if song_dict.get(song_id) is None:
        song_dict[song_id] = len(song_dict)

song_f = open("../dataset/members.csv", 'r')
lines = song_f.readlines()[1:]
for i, line in enumerate(lines):
    tokens = line.strip().split(',')
    msno = tokens[0]
    if user_dict.get(msno) is None:
        user_dict[msno] = len(user_dict)

def load_data(train, test, song):

    train = pd.read_csv(train)
    test = pd.read_csv(test)
    song = pd.read_csv(song)

    song_cols = ['song_id', 'artist_name', 'genre_ids', 'composer', 'lyricist', 'language']
    train = train.merge(song[song_cols], on='song_id', how='left')
    test = test.merge(song[song_cols], on='song_id', how='left')
    
    del song; gc.collect();

    train_user = train['msno'].astype('category')
    train_song = train['song_id'].astype('category')
    train_sst = train['source_system_tab'].astype('category')
    train_ssn = train['source_screen_name'].astype('category')
    train_st = train['source_type'].astype('category')
    train_target = train['target'].astype(np.uint8)

    test_user = test['msno'].astype('category')
    test_sst = test['source_system_tab'].astype('category')
    test_ssn = test['source_screen_name'].astype('category')
    test_st = test['source_type'].astype('category')
    test_idx = test['id'].astype(np.uint8)
    test_genre = test['genre_ids'].astype('category')
    test_artist = test['artist_name'].astype('category')

    for user in train_user:
        if user_dict.get(user) is None:
            user_dict[user] = len(user_dict)
    train_user = [user_dict.get(user) for user in train_user]
    for song in train_song:
        if song_dict.get(song) is None:
            song_dict[song] = len(song_dict)
    train_song = [song_dict.get(song) for song in train_song]
    for sst in train_sst:
        if sst_dict.get(sst) is None:
            sst_dict[sst] = len(sst_dict)
    train_sst = [sst_dict.get(sst) for sst in train_sst]
    for ssn in train_ssn:
        if ssn_dict.get(ssn) is None:
            ssn_dict[ssn] = len(ssn_dict)
    train_ssn = [ssn_dict.get(ssn) for ssn in train_ssn]
    for st in train_st:
        if st_dict.get(st) is None:
            st_dict[st] = len(st_dict)
    train_st = [st_dict.get(st) for st in train_st]

    train = train.drop_duplicates(['song_id'])
    tune_song = train['song_id'].astype('category')
    tune_genre = train['genre_ids'].astype('category')
    tune_artist = train['artist_name'].astype('category')
    tune_composer = train['composer'].astype('category')
    tune_lyricist = train['lyricist'].astype('category')
    tune_language = train['language'].astype('category')

    for song in tune_song:
        if song_dict.get(song_id) is None:
            song_dict[song] = len(song_dict)
    tune_song = [0 if song_dict.get(song) is None else song_dict.get(song) for song in tune_song]
    for genre in tune_genre:
        if genre_dict.get(genre) is None:
            genre_dict[genre] = len(genre_dict)
    tune_genre = [0 if genre_dict.get(genre) is None else genre_dict.get(genre) for genre in tune_genre]
    for artist in tune_artist:
        if artist_dict.get(artist) is None:
            artist_dict[artist] = len(artist_dict)
    tune_artist = [0 if artist_dict.get(artist) is None else artist_dict.get(artist) for artist in tune_artist]

    del train; gc.collect();

    test_user = [0 if user_dict.get(user) is None else user_dict.get(user) for user in test_user]
    test_genre = [0 if genre_dict.get(genre) is None else genre_dict.get(genre) for genre in test_genre]
    test_artist = [0 if artist_dict.get(artist) is None else artist_dict.get(artist) for artist in test_artist]
    test_sst = [0 if sst_dict.get(sst) is None else sst_dict.get(sst) for sst in test_sst]
    test_ssn = [0 if ssn_dict.get(ssn) is None else ssn_dict.get(ssn) for ssn in test_ssn]
    test_st = [0 if st_dict.get(st) is None else st_dict.get(st) for st in test_st]

    del test; gc.collect();

    return len(user_dict), len(song_dict), len(genre_dict), len(artist_dict), len(sst_dict), len(ssn_dict), len(st_dict), train_user, train_song, train_sst, train_ssn, train_st, train_target, tune_song, tune_genre, tune_artist, test_idx, test_user, test_genre, test_artist, test_sst, test_ssn, test_st

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
