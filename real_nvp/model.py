"""
The core Real-NVP model
"""

import tensorflow as tf
import pixel_cnn_pp.scopes as scopes
from pixel_cnn_pp.scopes import add_arg_scope
import real_nvp.nn as nn



layers = []
def construct_model_spec():
  layers.append(nn.CouplingLayer('checkerboard0', name='Coupling1'))
  layers.append(nn.CouplingLayer('checkerboard1', name='Coupling2'))    


@add_arg_scope
def model_spec(x, init=True, ema=None):
  counters = {}
  #with scopes.arg_scope([],counters=counters,init=init, ema=ema):
  xs = nn.int_shape(x)
  sum_log_det_jacobians = tf.zeros(xs[0])    

  # corrupt data (Tapani Raiko's dequantization)
  y = x*0.5 + 0.5
  y = y*255.0
  corruption_level = 1.0
  y = y + corruption_level * tf.random_uniform(xs)
  y = y/(255.0 + corruption_level)

  #model logit instead of the x itself    
  alpha = 1e-5
  y = y*(1-alpha) + alpha*0.5
  jac = tf.reduce_sum(-tf.log(y) - tf.log(1-y), [1,2,3])
  y = tf.log(y) - tf.log(1-y)
  sum_log_det_jacobians += jac

  if len(layers) == 0:
    construct_model_spec()

  # construct forward pass    
  for layer in layers:
    y,jac = layer.coupling_layer(y)
    sum_log_det_jacobians += jac        

  return y,sum_log_det_jacobians

def inv_model_spec(y):
  # construct inverse pass for sampling
  x = y
  for layer in reversed(layers):
    x = layer.inv_coupling_layer(x)
    
  # inverse logit
  x = tf.inv(1 + tf.exp(-x))

  # scale to [-1,1]
  #x = (x-0.5)*2.0

  return x
    
  

@add_arg_scope
def simple_model_spec(x, init=True, ema=None):
  
  counters = {}
  with scopes.arg_scope([],counters=counters,init=init, ema=ema):
    xs = nn.int_shape(x)
    layer = tf.reshape(tf.slice(x, [0, 0], [-1, 1]), [-1, 1, 1])
    print ("layer shape:")
    print (nn.int_shape(layer))
    channel = 100
    padding = 'SAME'
    weights = tf.get_variable("weights_1", [1, 1, channel], tf.float32, tf.contrib.layers.xavier_initializer())
    layer = tf.nn.conv1d(layer, weights, 1, padding=padding)
    biases = tf.get_variable("biases_1", [channel,], tf.float32, tf.zeros_initializer)
    layer = tf.nn.bias_add(layer, biases)
    layer = tf.nn.relu(layer)
    weights = tf.get_variable("weights_2", [1, channel, channel], tf.float32, tf.contrib.layers.xavier_initializer())
    layer = tf.nn.conv1d(layer, weights, 1,  padding=padding)
    biases = tf.get_variable("biases_2", [channel,], tf.float32, tf.zeros_initializer)
    layer = tf.nn.bias_add(layer, biases)
    layer = tf.nn.relu(layer)
    weights = tf.get_variable("weights_3", [1, channel, channel], tf.float32, tf.contrib.layers.xavier_initializer())
    layer = tf.nn.conv1d(layer, weights, 1,  padding=padding)
    biases = tf.get_variable("biases_3", [channel,], tf.float32, tf.zeros_initializer)
    layer = tf.nn.bias_add(layer, biases)
    layer = tf.nn.relu(layer)

    weights = tf.get_variable("weights_4", [1, channel, channel], tf.float32, tf.contrib.layers.xavier_initializer())
    layer = tf.nn.conv1d(layer, weights, 1,  padding=padding)
    biases = tf.get_variable("biases_4", [channel,], tf.float32, tf.contrib.layers.xavier_initializer())
    layer = tf.nn.bias_add(layer, biases)
    layer = tf.nn.relu(layer)

    weights = tf.get_variable("weights_6", [1, channel, 1], tf.float32, tf.contrib.layers.xavier_initializer())
    layer = tf.nn.conv1d(layer,weights, 1,  padding=padding)
    out_layer = layer
    return out_layer
