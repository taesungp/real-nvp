import tensorflow as tf
import numpy as np


def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print ("[%2d] %s %s = %s" % (idx, op.name, shape, count))
    total_count += int(count)
  print ("[Total] variable size: %s" % "{:,}".format(total_count))
