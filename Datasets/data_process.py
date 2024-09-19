'''
This file is used for data processing of SimGCL/XSimGCL
'''
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

data = 'beer'

def loaddata(filename):
    with open(filename, 'rb') as fs:
        ret = (pickle.load(fs) != 0).astype(np.float32)
    if type(ret) != coo_matrix:
        ret = sp.coo_matrix(ret)
    return ret

def generate(filename, data):
    left, right = data.nonzero()
    with open(filename, 'w') as fn:
        for i,j in zip(left, right):
            fn.write(str(i) + ' ' + str(j) + ' ' + str(1) + '\n')


if data == 'yelp':
    predir = 'sparse_yelp/'
elif data == 'lastfm':
    predir = 'lastFM/'
elif data == 'beer':
    predir = 'beerAdvocate/'
predir = predir
trnfile = predir + 'trnMat.pkl'
tstfile = predir + 'tstMat.pkl'

trndata = loaddata(trnfile)
tstdata = loaddata(tstfile)
generate('train.txt', trndata)
generate('test.txt', tstdata)

