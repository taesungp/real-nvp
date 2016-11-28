import numpy as np
import tensorflow as tf

def int_shape(x):
    return list(map(int, x.get_shape()))

class Layer():
  def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
    raise NotImplementedError(str(type(self)))

  def backward(self, y, z):
    raise NotImplementedError(str(type(self)))

class CouplingLayer(Layer):

  # |use_batchnorm|: whether using batch norm at the very output of l,m.
  # Batchnorm is used for all other convolutions regardless of this flag
  def __init__(self, mask_type, hidden_dim=64, use_batchnorm = True, name='Coupling'):
    self.mask_type = mask_type
    self.name = name
    self.hidden_dim=hidden_dim

    # for batch normalization using moving average
    self.use_batchnorm = use_batchnorm
    self.mu_l = tf.zeros([1], name=name+"/mu_l")
    self.sig2_l = tf.ones([1], name=name+"/sig2_l")
    self.mu_m = tf.zeros([1], name=name+"/mu_m")
    self.sig2_m = tf.ones([1], name=name+"/sig2_m")
    self.momentum = 0.5

  def batch_norm(self, x):
    mu = tf.reduce_mean(x)
    sig2 = tf.reduce_mean(tf.square(x-mu))    
    x = (x-mu)/tf.sqrt(sig2 + 1.0e-5)
    return x, sig2

  def get_normalized_weights(self, name, weights_shape):
    weights = tf.get_variable(name, weights_shape, tf.float32,
                              tf.contrib.layers.xavier_initializer())
    scale = tf.get_variable(name + "_scale", [1], tf.float32, 
                              tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(5e-5))
    norm = tf.sqrt(tf.reduce_sum(tf.square(weights))+1e-5)
    return weights/norm * scale
    

  
  # corresponds to the function m and l in the RealNVP paper
  # returns l,m
  def function_l_m(self,x,mask,name='function_l_m'):
    with tf.variable_scope(name):
      channel = self.hidden_dim
      padding = 'SAME'
      xs = int_shape(x)
      kernel_h = 3
      kernel_w = 3
      input_channel = xs[3]
      x = tf.check_numerics(x, "x has NaN")
      y = x

      y,_ = self.batch_norm(y)
      weights_shape = [1, 1, input_channel, channel]
      weights = self.get_normalized_weights("weights_input", weights_shape)
      
      y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
      y,_ = self.batch_norm(y)
      y = tf.nn.relu(y)

      skip = y
      num_residual_blocks = 8
      for r in range(num_residual_blocks):        
        weights_shape = [kernel_h, kernel_w, channel, channel]
        weights = self.get_normalized_weights("weights%d_1" % r, weights_shape)
        y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
        y,_ = self.batch_norm(y)
        y = tf.nn.relu(y)
        weights_shape = [kernel_h, kernel_w, channel, channel]
        weights = self.get_normalized_weights("weights%d_2" % r, weights_shape)
        y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
        y,_ = self.batch_norm(y)
        y += skip
        y = tf.nn.relu(y)
        skip = y

        
      weights = self.get_normalized_weights("weights_output", [1, 1, channel, input_channel*2])
      y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)    

      
      y = tf.tanh(y)
      scale_factor = tf.get_variable("weights_tanh_scale", [1], tf.float32, tf.contrib.layers.xavier_initializer())
      y *= scale_factor
      y = tf.check_numerics(y, "y has NaN")


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
    elif 'channel' in mask_type:
      white = tf.ones([xs[0], xs[1], xs[2], xs[3]//2])
      black = tf.zeros([xs[0], xs[1], xs[2], xs[3]//2])
      if mask_type == 'channel0':
        b = tf.concat(3, [white, black])
      else:
        b = tf.concat(3, [black, white])

    bs = int_shape(b)
    assert bs == xs

    return b

  # corresponds to the coupling layer of the RealNVP paper
  # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
  # log_det_jacobian is a 1D tensor of size (batch_size)
  def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
    with tf.variable_scope(self.name):
      xs = int_shape(x)
      b = self.get_mask(xs, self.mask_type)

      # masked half of x
      x1 = x * b
      l,m = self.function_l_m(x1, b)
      y = x1 + tf.mul(-b+1.0, x*tf.check_numerics(tf.exp(l), "exp has NaN") + m)
      log_det_jacobian = tf.check_numerics(tf.reduce_sum(l, [1,2,3]), "l has NaN")
      sum_log_det_jacobians += log_det_jacobian

      #y,sig2 = self.batch_norm(y)
      #sum_log_det_jacobians += -xs[1]*xs[2]*xs[3]*0.5*tf.log(sig2 + 1e-6)
      

      #log_det_jacobian += xs[1]*xs[2]*xs[3]*0.5*tf.log(tf.sqrt(self.sig2_l + 1e-5)*10)
      

      return y,sum_log_det_jacobians, z

  def backward(self, y, z):    
    with tf.variable_scope(self.name, reuse=True):
      ys = int_shape(y)
      b = self.get_mask(ys, self.mask_type)

      y1 = y * b
      l,m = self.function_l_m(y1, b)
      x = y1 + tf.mul( y*(-b+1.0) - m, tf.check_numerics(tf.exp(-l), "exp inv has Nan"))
      return x, z

class SqueezingLayer(Layer):
  def __init__(self, name="Squeeze"):
    self.name = name

  def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
    xs = int_shape(x)
    assert xs[1] % 2 == 0 and xs[2] % 2 == 0
    #y = tf.reshape(x, [xs[0], xs[1]//2, xs[2]//2, xs[3]*4])
    y = tf.space_to_depth(x, 2)
    if z is not None:
      z = tf.space_to_depth(z, 2)      

    return y,sum_log_det_jacobians, z

  def backward(self, y, z):
    ys = int_shape(y)
    assert ys[3] % 4 == 0
    #x = tf.reshape(y, [ys[0], ys[1]*2, ys[2]*2, ys[3]//4])
    x = tf.depth_to_space(y,2)

    if z is not None:
      z = tf.depth_to_space(z,2)

    return x, z

class FactorOutLayer(Layer):
  def __init__(self, scale, name='FactorOut'):
    self.scale = scale
    self.name = name
  
  def forward_and_jacobian(self, x, sum_log_det_jacobians, z):

    xs = int_shape(x)
    split = xs[3]//2

    new_z = x[:,:,:,:split]
    x = x[:,:,:,split:]

    if z is not None:
      z = tf.concat(3, [z, new_z])
    else:
      z = new_z
    
    return x, sum_log_det_jacobians, z
  
  def backward(self, y, z):

    # At scale 0, 1/2 of the original dimensions are factored out
    # At scale 1, 1/4 of the original dimensions are factored out
    # ....
    # At scale s, (1/2)^(s+1) are factored out
    # Hence, at backward pass of scale s, (1/2)^(s) of z should be factored in
    
    zs = int_shape(z)
    split = zs[3] // (2**self.scale)
    new_y = z[:,:,:,-split:]
    z = z[:,:,:,:-split]

    assert (int_shape(new_y)[3] == split)

    if y is not None:
      x = tf.concat(3, [new_y, y])
    else:
      x = new_y

    return x, z
            
    
    



# Given the output of the network and all jacobians, 
# compute the log probability. 
# Equation (3) of the RealNVP paper
def compute_log_prob_x(z, sum_log_det_jacobians):
  
  # y is assumed to be in standard normal distribution
  # 1/sqrt(2*pi)*exp(-0.5*x^2)
  zs = int_shape(z)
  K = zs[1]*zs[2]*zs[3] #dimension of the Gaussian distribution

  log_density_z = -0.5*tf.reduce_sum(tf.check_numerics(tf.square(z), "square z"), [1,2,3]) - 0.5*K*np.log(2*np.pi)

  log_density_x = tf.check_numerics(log_density_z, "log_density_z") + \
                  tf.check_numerics(sum_log_det_jacobians, "sum_log_det_jacobians")

  # to go from density to probability, one can 
  # multiply the density by the width of the 
  # discrete probability area, which is 1/256.0, per dimension
  log_prob_x = log_density_x - K*tf.log(256.0)

  return tf.check_numerics(log_prob_x, "log_prob_x has NaN")


def loss(z, sum_log_det_jacobians, logits_fake=None, logits_real=None):
  loss_gaussianization = -tf.reduce_sum(compute_log_prob_x(z, sum_log_det_jacobians))

  if logits_fake is not None and logits_real is not None:
    #loss_adversarial = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.zeros_like(logits_fake)))

    #loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.ones_like(logits_real)))
    #loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.zeros_like(logits_fake)))

    #loss_adversarial = -loss_real + -loss_fake
    loss_adversarial = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.ones_like(logits_fake)))
    


    zs = int_shape(z)
    K = zs[0]*zs[1]*zs[2]*zs[3]*np.log(2.)
    gamma = 0.01

    combined_loss = loss_gaussianization + gamma*K*loss_adversarial
    #combined_loss = K*loss_adversarial
  else:
    combined_loss = loss_gaussianization  
    
  return combined_loss



  
  
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


