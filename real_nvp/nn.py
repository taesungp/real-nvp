import numpy as np
import tensorflow as tf

def int_shape(x):
    return list(map(int, x.get_shape()))


class CouplingLayer():
  def __init__(self, mask_type, name='Coupling'):
    self.mask_type = mask_type
    self.name = name
  
  # corresponds to the function m and l in the RealNVP paper
  # returns l,m
  def function_l_m(self,x,mask,name='function_l_m'):
    with tf.variable_scope(name):
      channel = 16
      padding = 'SAME'
      xs = int_shape(x)
      kernel_h = 3
      kernel_w = 3
      input_channel = xs[3]

      weights_shape = [kernel_h, kernel_w, input_channel, channel]
      weights = tf.get_variable("weights", weights_shape, tf.float32, 
                            tf.contrib.layers.xavier_initializer())
      y = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding=padding)
      y = tf.nn.relu(y)

      weights = tf.get_variable("weights_output", [1, 1, channel, input_channel*2],
                                tf.float32, tf.contrib.layers.xavier_initializer())
      y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)    

      l = y[:,:,:,:input_channel] * (-mask+1)
      m = y[:,:,:,input_channel:] * (-mask+1)

      return l,m


  # returns constant tensor of masks
  # |xs| is the size of tensor
  # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
  # |b| has the dimension of |xs|
  def get_mask(self, xs, mask_type):

    if 'checkerboard' in mask_type:
      unit0 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
      unit1 = -unit0 + 1.0
      unit = unit0 if mask_type == 'checkerboard0' else unit1
      unit = tf.reshape(unit, [1, 2, 2, 1])

    b = tf.tile(unit, [xs[0], xs[1]//2, xs[2]//2, xs[3]])

    assert 'checkerboard' in mask_type


    return b

  # corresponds to the coupling layer of the RealNVP paper
  # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
  # log_det_jacobian is a 1D tensor of size (batch_size)
  def coupling_layer(self, x):
    with tf.variable_scope(self.name):
      xs = int_shape(x)
      b = self.get_mask(xs, self.mask_type)

      # masked half of x
      x1 = x * b
      l,m = self.function_l_m(x1, b)
      y = x1 + tf.mul(-b+1.0, x*tf.check_numerics(tf.exp(l), "exp has NaN") + m)
      log_det_jacobian = tf.check_numerics(tf.reduce_sum(l, [1,2,3]), "l has NaN")

      return y,log_det_jacobian

  def inv_coupling_layer(self, y):    
    with tf.variable_scope(self.name, reuse=True):
      ys = int_shape(y)
      b = self.get_mask(ys, self.mask_type)

      y1 = y * b
      l,m = self.function_l_m(y1, b)
      x = y1 + tf.mul( y*(-b+1.0) - m, tf.exp(-l))
      return x

# Given the output of the network and all jacobians, 
# compute the log probability. 
# Equation (3) of the RealNVP paper
def compute_log_prob_x(z, sum_log_det_jacobians):
  
  # y is assumed to be in standard normal distribution
  # 1/sqrt(2*pi)*exp(-0.5*x^2)
  zs = int_shape(z)
  K = zs[1]*zs[2]*zs[3] #dimension of the Gaussian distribution
  print (0.5*K*np.log(2*np.pi))

  log_density_z = -0.5*tf.reduce_sum(tf.square(z), [1,2,3]) - 0.5*K*np.log(2*np.pi)

  log_density_x = log_density_z + sum_log_det_jacobians

  # to go from density to probability, one can 
  # multiply the density by the width of the 
  # discrete probability area, which is 1/256.0, per dimension
  log_prob_x = log_density_x - K*tf.log(256.0)

  return tf.check_numerics(log_prob_x, "log_prob_x has NaN")


def loss(z, sum_log_det_jacobians):
  return -tf.reduce_sum(compute_log_prob_x(z, sum_log_det_jacobians))


  
  
def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


