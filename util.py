import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import rcParams

def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print ("[%2d] %s %s = %s" % (idx, op.name, shape, count))
    total_count += int(count)
  print ("[Total] variable size: %s" % "{:,}".format(total_count))


def save_images_with_nll(images, nlls):
  num_images = images.shape[0]
  num_images_per_row = 4
  num_images_per_column = (num_images + num_images_per_row - 1) // num_images_per_row
  idx = 0
  for i in range(num_images_per_column):
    for j in range(num_images_per_row):
      plt.subplot2grid((num_images_per_column,num_images_per_row),(i, j))
      plt.axis('off')
      plt.imshow(images[idx])
      plt.title('%f' % nlls[idx])      
      idx += 1
      if idx >= num_images:
        plt.savefig('test_results/samples_%s.png' % time.strftime("%m_%d_%H_%M_%S"), bbox_inches='tight')
        return
  

