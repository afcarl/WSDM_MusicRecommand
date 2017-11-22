import numpy as np

mem2id = {}
id2mem = []

mem_f = open("../dataset/members.csv", "r")
lines = mem_f.readlines()

for i, line in enumerate(lines):
    if i == 0:
        continue

    tokens = line.strip().split(',')
    msno = tokens[0]

    if mem2id.get(msno) == None:
        mem2id[msno] = len(id2mem)
        id2mem.append(msno)

np.save("npy/allmem2id.npy", mem2id)
np.save("npy/id2allmem.npy", id2mem)
