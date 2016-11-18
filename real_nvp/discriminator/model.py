# Discriminator of GAN
# from the implementation of carpedm20 on github. 

import tensorflow as tf
from real_nvp.discriminator.nn import *

def model_spec(x, init=True, ema=None):

  df_dim = 64
  batch_size = int_shape(x)[0]
  
  # batch normalization : deals with poor initialization helps gradient flow
  d_bn1 = batch_norm(name='d_bn1')
  d_bn2 = batch_norm(name='d_bn2')
  d_bn3 = batch_norm(name='d_bn3')
  
  
  h0 = lrelu(conv2d(x, df_dim, name='d_h0_conv'))
  h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
  h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
  h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
  h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')
  
  # h4 is the logit distribution
  return h4

  
