#!/usr/bin/python3

####################
# MODULES
####################

# Data wrangling
import numpy as np
import pandas as pd
import collections

# Clustering
import hdbscan

# Evaluation 
import time

####################
# FUNCTIONS
####################

# MNIST file reader
""" A function that can read MNIST's idx file format into numpy arrays.
    The MNIST data files can be downloaded from here:
    
    http://yann.lecun.com/exdb/mnist/
    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.
"""

import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


####################
# EXECUTION
####################

# Read in data
raw_train = read_idx("./mnist/train-images-idx3-ubyte")
train_data = np.reshape(raw_train, (60000,28*28))
train_labels =  read_idx("./mnist/train-labels-idx1-ubyte")
MNIST_data = train_data #[train_labels == chosen_number]

# Cluster the data
start_time = time.time()
hdbscan_labels = hdbscan.HDBSCAN(core_dist_n_jobs = 24).fit_predict(MNIST_data)
collections.Counter(hdbscan_labels)
print("--- %s seconds ---" % (time.time() - start_time))
