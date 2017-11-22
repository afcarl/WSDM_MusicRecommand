import csv
import operator 
import numpy as np

path_dir = '/home/yongqyu/Works/MusicRec/dataset/'

lines=[]
user2song={} 
song2user={} 

with open(path_dir+'filtered_train.csv') as f:
    csvReader = csv.reader(f)
    for i, line in enumerate(csvReader):
        if i == 0:
            continue
        msno = line[0]
        song = line[1]
        if user2song.get(msno) == None:
            user2song[msno] = [song]
        else:
            user2song.get(msno).append(song)

        if song2user.get(song) == None:
            song2user[song] = [msno]
        else:
            song2user.get(song).append(msno)

np.save("npy/user2song.npy", user2song)
np.save("npy/song2user.npy", song2user)

'''
with open(path_dir+"members.csv") as tf:
    with open("user_song.csv", 'wb') as wf:
        wr = csv.writer(wf, delimiter=',')
        wr.writerow(['msno','song_id'])

        churnReader = csv.reader(tf)
        cnt = 0
        for i, line in enumerate(churnReader):
            if i == 0:
                continue
            msno = line[0]
            if user_dict.get(msno) == None:
                cnt += 1
            else:
                wr.writerow([msno, user_dict.get(msno)])
        print cnt
'''
