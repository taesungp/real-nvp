"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12,22
import datetime
import dateutil.tz
import scipy.misc
from scipy import misc,ndimage
from skimage import io, color
import time

import pixel_cnn_pp.nn as pixel_cnn_nn
import real_nvp.nn as real_nvp_nn
import pixel_cnn_pp.plotting as plotting
import pixel_cnn_pp.scopes as scopes
from pixel_cnn_pp.model import model_spec as pixel_cnn_model_spec
from real_nvp.model import model_spec as real_nvp_model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
import data.csv_data as csv_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=int, default=0, help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
# model
parser.add_argument('--model', type=str, default='real_nvp', help='model name: pixel_cnn, real_nvp')
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=192, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--sample_batch_size', type=int, default=16, help='How many images to process in paralell during sampling?')
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
DataLoader = {'cifar':cifar10_data.DataLoader, 
              'imagenet':imagenet_data.DataLoader,
              'csv':csv_data.DataLoader
            }[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
model_spec = {'pixel_cnn': pixel_cnn_model_spec,
              'real_nvp': real_nvp_model_spec}[args.model]
nn = {'pixel_cnn': pixel_cnn_nn,
      'real_nvp': real_nvp_nn}[args.model]

# create the model
model_opt = {'pixel_cnn':{ 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'dropout_p': args.dropout_p },
             'real_nvp':{}}[args.model]
model = tf.make_template('model', model_spec)

x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
# run once for data dependent initialization of parameters
gen_par = model(x_init, init=True, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# sample from the model
x_sample = tf.placeholder(tf.float32, shape=(args.sample_batch_size, ) + obs_shape)
model_opt_for_sampling = model_opt
if args.model == 'pixel_cnn':
  model_opt_for_sampling['dropout_p'] = 0
  gen_par = model(x_sample, ema=None, **model_opt_for_sampling)
  new_x_gen = nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix)
else:
  new_x_gen = model(x_sample, ema=None, **model_opt)
  #new_x_gen = nn.sample_from_gaussian(new_x_gen)  
def sample_from_model(sess):
  x_gen = np.zeros((args.sample_batch_size,) + obs_shape, dtype=np.float32)
  new_x_gen_np = sess.run(new_x_gen, {x_sample: x_gen})
  return new_x_gen_np

# get loss gradients over multiple GPUs
xs = []
grads = []
loss_gen = []
loss_gen_test = []
for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape))
    with tf.device('/gpu:%d' % i):
        # train
        gen_par,jacs = model(xs[i], ema=None, **model_opt)
        loss_gen.append(nn.loss(gen_par, jacs))
        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))
        # test
        model_opt_for_testing = model_opt
        gen_par,jacs = model(xs[i], ema=None, **model_opt)
        loss_gen_test.append(nn.loss(gen_par, jacs))

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
def prepro(x):
  x[:,:,:,0] = 255.0
  x[:,:,:,1] = np.random.uniform(0.0,0.5)*255.0
  x[:,:,:,1] = 0.0
  x[:,:,:,2] = 0.0

  return np.cast[np.float32]((x - 127.5) / 127.5)

def create_montage_image(images, height, width, n_row, n_col, n_channel):
  images = images.reshape((n_row, n_col, height, width, n_channel))
  images = images.transpose(0, 2, 1, 3, 4)
  images = images.reshape((height * n_row, width * n_col, n_channel))
  if n_channel == 1:
    images = np.tile(images, (1,1,3))
  return images


def compute_likelihood(xf):
  print ("computing likelihood of image with mean %f" % np.mean(xf))
  xfs = np.split(xf, args.nr_gpu)
  #print ("Splitting %d images into groups of size %d" % (xf.shape[0], xfs.shape[0]))
  feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
  l = sess.run(bits_per_dim_test, feed_dict)
  return l

# |x| is assumed to be in range [-1,1]
# distortions is a dictinary of distortions
def distort_images(x, distortions):
  x = x*0.5 + 0.5
  x = np.clip(x, 0.0, 1.0)
  num_images = x.shape[0]
  width = x.shape[1]
  height = x.shape[2]
  
  distorted = np.copy(x)
  
  distorted = color.rgb2lab(distorted)
  for i in range(num_images):
    distorted[i,:,:,1] += distortions['tint_a']
    distorted[i,:,:,2] += distortions['tint_b']
    distorted[i] = color.lab2rgb(distorted[i])

  distorted = color.rgb2lab(distorted)
  distorted[:,:,:,1:3] *= distortions['saturation_scale']


  for i in range(distorted.shape[0]):
    distorted[i] = color.lab2rgb(distorted[i])

      
  distorted = 0.5 + (distorted - 0.5) * distortions['contrast_scale']
  distorted = np.clip(distorted, 0.0, 1.0)
      
  
  distorted += (np.random.randn(num_images,width,height,3)-0.5)*distortions['noise_scale']
  distorted = np.clip(distorted, 0.0, 1.0)

  

  if distortions['apply_gradient'] == True:
    top_color = np.array([0.0, 0.0, 0.0])
    bottom_color = np.array([0.8, 0.2, 0.2])
    height = distorted.shape[1]
    for i in range(height):
      grad_color = top_color * float(height - i) / height + bottom_color * float(i)/height
      distorted[:,i,:,0] += grad_color[0]
      distorted[:,i,:,1] += grad_color[1]
      distorted[:,i,:,2] += grad_color[2]
      distorted = np.clip(distorted,0.0,1.0)
  
  distorted = ndimage.filters.gaussian_filter(distorted, distortions['blur'])

  if distortions['transform'] == True:
    distorted[:4,:,:,:] = predict(distorted[:4,:,:,:], x[:4,:,:,:])
    #sample_x = sample_from_model(sess)
    #sample_x = np.tile(sample_x, (2, 1, 1, 1))
    #distorted = distorted*0.5 + 0.5
  
  return (distorted - 0.5)*2.0


def test_likelihoods(x):
  #x = x[:4,:,:,:]
  distortions = {}
  distortions['tint_a'] = 0
  distortions['tint_b'] = 0
  distortions['saturation_scale'] = 1.0
  distortions['contrast_scale'] = 1.0
  distortions['noise_scale'] = 0.0
  distortions['apply_gradient'] = False
  distortions['blur'] = 0.0
  distortions['transform'] = False
  radius = 2
  w = x.shape[1]
  h = x.shape[2]
  num_images_per_row = radius*2+1
  nll_matrix = np.zeros((num_images_per_row,num_images_per_row))
  for a in range(-radius,radius+1):
    for b in range(-radius,radius+1):
      #distortions['tint_a'] = a*4
      #distortions['tint_b'] = b*4
      distortions['saturation_scale'] = 1.0 + a*0.3
      distortions['contrast_scale'] = 1.0 + b*0.3
      #distortions['noise_scale'] = 0.03*a
      #distortions['apply_gradient'] = True if b > 0 else False
      #distortions['blur'] = 0.15 * (b + radius)
      #distortions['transform'] = True if b > 0 else False            
      images = distort_images(x, distortions)        
      print ("distort_images changed images of batch size %d to %d" % (x.shape[0], images.shape[0]))
      nll = compute_likelihood(images)
      nll_matrix[a+radius,b+radius] = nll
      print ("computing likelihood of (a,b)=(%d,%d) = %f..." % (a,b,nll))
      images = (images*0.5 + 0.5)
      images = images[:4,:,:,:]
      montage = create_montage_image(images,w,h,2,2,3)
      plt.subplot2grid((num_images_per_row+2,num_images_per_row),(a+radius,b+radius))
      plt.axis('off')
      plt.imshow(montage)
      plt.title('%f' % nll)
  
  chart = plt.subplot2grid((num_images_per_row+2,num_images_per_row),(num_images_per_row,0), 
                           colspan=num_images_per_row)
  chart.plot(range(-radius, radius+1), nll_matrix)
  plt.ylabel("nll")
  plt.title("nll by the vertical axis")
    

  chart = plt.subplot2grid((num_images_per_row+2,num_images_per_row),(num_images_per_row+1,0), 
                           colspan=num_images_per_row)
  chart.plot(range(-radius, radius+1), np.transpose(nll_matrix))
  plt.ylabel("nll")
  plt.title("nll by the horizontal axis")


  plt.savefig('test/test_contrast_and_saturation_%s.png' % time.strftime("%m_%d_%H_%M_%S"), bbox_inches='tight')


# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate
with tf.Session() as sess:
    for epoch in range(args.max_epochs):
        begin = time.time()

        # init
        if epoch == 0:
            x = train_data.next(args.init_batch_size) # manually retrieve exactly init_batch_size examples
            train_data.reset() # rewind the iterator back to 0 to do one full epoch
            print('initializing the model...')
            sess.run(initializer,{x_init: prepro(x)})
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)

        # train for one epoch
        train_losses = []
        skip_train = False
        if not skip_train:
          print ("Training (%d/%d) started" % (epoch, args.max_epochs))
          for t,x in enumerate(train_data):          
            # prepro the data and split it for each gpu
            xf = prepro(x)
            xfs = np.split(xf, args.nr_gpu)
            #print (xfs)
            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            feed_dict = { tf_lr: lr }
            feed_dict.update({ xs[i]: xfs[i] for i in range(args.nr_gpu) })
            l,_ = sess.run([bits_per_dim, optimizer], feed_dict)
            train_losses.append(l)
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test split
        test_losses = []
        skip_test = False
        if not skip_test:
          print ("Testing...")
          for x in test_data:
            xf = prepro(x)
            #test_likelihoods(xf)
            xfs = np.split(xf, args.nr_gpu)
            feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
            l = sess.run(bits_per_dim_test, feed_dict)
            test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            print ("Generating samples...")

            # generate samples from the model
            #sample_x = sample_from_model(sess)
            #img_tile = plotting.img_tile(sample_x, aspect_ratio=1.0, border_color=1.0, stretch=True)
            #img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            #plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d.png' % (args.data_set, epoch)))
            #plotting.plt.close('all')
            
            # save params
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))
