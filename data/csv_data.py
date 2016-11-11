import os
import sys
import numpy as np

class DataLoader(object):
    """ an object that loads data from .csv file """

    def __init__(self, data_path, subset, batch_size, rng=None, shuffle=False):
        """ 
        - data_dir is location of .csv file
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        # load CSV file
        self.data = np.loadtxt(open(self.data_path,"rb"),delimiter=",",skiprows=1)
        print ("size of csv file")
        print (self.data.shape)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]        
        self.p += self.batch_size
        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)



