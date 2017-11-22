import numpy as np

user2cnt = {}
song2cnt = {}

user_thr = 5
song_thr = 5

song_f = open("../dataset/songs.csv", "r")
song_extra_f = open("../dataset/song_extra_info.csv", "r")
memb_f = open("../dataset/members.csv", "r")
train_f = open("../dataset/train.csv", "r")

lines = train_f.readlines()

for i, line in enumerate(lines):
    if i == 0:
        continue

    tokens = line.strip().split(',')
    msno = tokens[0]
    song_id = tokens[1]

    if user2cnt.get(msno) == None:
        user2cnt[msno] = [song_id] 
    else:
        user2cnt.get(msno).append(song_id)

    if song2cnt.get(song_id) == None:
        song2cnt[song_id] = [msno] 
    else:
        song2cnt.get(song_id).append(msno)
print "Complete to make dictionary"

while(1):
    del_cnt = 0
    for user, songs in user2cnt.items():
        if len(songs) < user_thr:
            for song in songs:
                del song2cnt.get(song)[song2cnt.get(song).index(user)]
                del_cnt += 1
            del user2cnt[user]

    for song, users in song2cnt.items():
        if len(users) < song_thr:
            for user in users:
                del user2cnt.get(user)[user2cnt.get(user).index(song)]
                del_cnt += 1
            del song2cnt[song]

    if del_cnt == 0:
        break
print "Complete to remove items under the threshold"

train_wf = open("../dataset/filtered_train.csv", 'w')
for i, line in enumerate(lines):
    if i == 0:
        train_wf.write(line)

    tokens = line.strip().split(',')
    msno = tokens[0]
    song_id = tokens[1]

    if msno in user2cnt and song_id in song2cnt:
        train_wf.write(line)
print "Complete to make filtered_train.csv"

lines = song_f.readlines()
train_wf = open("../dataset/filtered_songs.csv", 'w')
for i, line in enumerate(lines):
    if i == 0:
        train_wf.write(line)

    tokens = line.strip().split(',')
    song_id = tokens[0]

    if song_id in song2cnt:
        train_wf.write(line)
print "Complete to make filtered_songs.csv"

lines = song_extra_f.readlines()
train_wf = open("../dataset/filtered_song_extra_info.csv", 'w')
for i, line in enumerate(lines):
    if i == 0:
        train_wf.write(line)

    tokens = line.strip().split(',')
    song_id = tokens[0]

    if song_id in song2cnt:
        train_wf.write(line)
print "Complete to make filtered_songs.csv"

lines = memb_f.readlines()
train_wf = open("../dataset/filtered_members.csv", 'w')
for i, line in enumerate(lines):
    if i == 0:
        train_wf.write(line)

    tokens = line.strip().split(',')
    msno = tokens[0]

    if msno in user2cnt:
        train_wf.write(line)
print "Complete to make filtered_members.csv"
