import pickle

with open("data/ysf.pkl", "rb") as inp:
    data = pickle.load(inp)

import gensim

print(gensim.__path__)
