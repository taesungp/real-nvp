"""
The core Real-NVP model
"""

import tensorflow as tf
import pixel_cnn_pp.scopes as scopes
from pixel_cnn_pp.scopes import add_arg_scope
import real_nvp.nn as nn


@add_arg_scope
def model_spec(x, init=True, ema=None):
  counters = {}
  with scopes.arg_scope([],counters=counters,init=init, ema=ema):
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

    
    # coupling layers
    y,jac = nn.coupling_layer(y, 'checkerboard0', name='Coupling1')
    sum_log_det_jacobians += jac

    y,jac = nn.coupling_layer(y, 'checkerboard1', name='Coupling2')
    sum_log_det_jacobians += jac

    return y,sum_log_det_jacobians
    
  

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
