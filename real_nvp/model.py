"""
The core Real-NVP model
"""

import tensorflow as tf
import pixel_cnn_pp.scopes as scopes
from pixel_cnn_pp.scopes import add_arg_scope
import real_nvp.nn as nn



layers = []
def construct_model_spec():
  global layers
  num_scales = 2
  for scale in range(num_scales-1):    
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
    layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale))
    layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale))
    layers.append(nn.CouplingLayer('channel0', use_batchnorm=True, name='Channel%d_3' % scale))
    layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

  # final layer
  scale = num_scales-1
  layers.append(nn.CouplingLayer('checkerboard0', hidden_dim=128, name='Checkerboard%d_1' % scale))
  layers.append(nn.CouplingLayer('checkerboard1', hidden_dim=128, name='Checkerboard%d_2' % scale))
  layers.append(nn.CouplingLayer('checkerboard0', hidden_dim=128, name='Checkerboard%d_3' % scale))
  layers.append(nn.CouplingLayer('checkerboard1', hidden_dim=128, name='Checkerboard%d_4' % scale))
  layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))


  
final_latent_dimension = []

@add_arg_scope
def model_spec(x, init=True, ema=None):
  counters = {}
  #with scopes.arg_scope([],counters=counters,init=init, ema=ema):
  xs = nn.int_shape(x)
  sum_log_det_jacobians = tf.zeros(xs[0])    

  y = x*0.5+0.5

  #model logit instead of the x itself    
  alpha = 1e-5
  y = y*(1-alpha) + alpha*0.5
  jac = tf.reduce_sum(-tf.log(y) - tf.log(1-y), [1,2,3])
  y = tf.log(y) - tf.log(1-y)
  sum_log_det_jacobians += jac

  if len(layers) == 0:
    construct_model_spec()

  # construct forward pass    
  z = None
  for layer in layers:
    y,jac,z = layer.forward_and_jacobian(y, jac, z)

  z = tf.concat(3, [z,y])

  # record dimension of the final variable
  global final_latent_dimension
  final_latent_dimension = nn.int_shape(z)

  return z,jac

def inv_model_spec(y):
  # construct inverse pass for sampling
  shape = final_latent_dimension
  z = tf.reshape(y, [-1, shape[1], shape[2], shape[3]])
  y = None
  for layer in reversed(layers):
    y,z = layer.backward(y,z)
    
  x = y

  # inverse logit
  x = tf.inv(1 + tf.exp(-x))
  #alpha = 1e-5
  #x = (x-alpha*0.5)/(1-alpha)
  
  # scale to [-1,1]
  x = (x-0.5)*2.0
  x = tf.clip_by_value(x, -1.0, 1.0)
  
  

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
