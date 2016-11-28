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
from real_nvp.model import inv_model_spec as real_nvp_inv_model_spec
import real_nvp.discriminator.nn as dn
from real_nvp.discriminator.model import model_spec as discriminator_model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
import data.csv_data as csv_data
import util

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=4, help='Every how many epochs to write checkpoint/samples?')
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
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=50000, help='How many epochs to run in total?')
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
model = tf.make_template('generator_model', model_spec)
inv_model = tf.make_template('generator_model', real_nvp_inv_model_spec, 
                             unique_name_='generator_model')
discriminator_model = tf.make_template('discriminator_model', discriminator_model_spec)

x_init = tf.placeholder(tf.float32, shape=(args.batch_size,) + obs_shape)
# run once for data dependent initialization of parameters
gen_par,_ = model(x_init, init=True, **model_opt)
#discriminator_par = discriminator_model(gen_par, init=True, **model_opt)
discriminator_par = discriminator_model(x_init, init=True, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
#ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
#maintain_averages_op = tf.group(ema.apply(all_params))

# sample from the model
x_sample = tf.placeholder(tf.float32, shape=(args.sample_batch_size, ) + obs_shape)
new_x_gen = inv_model(x_sample)
#new_x_gen = nn.sample_from_gaussian(new_x_gen)  
def sample_from_model(sess):
  x_gen = np.random.normal(0.0, 1.0, (args.sample_batch_size,) + obs_shape)
  new_x_gen_np = sess.run(new_x_gen, {x_sample: x_gen})
  return new_x_gen_np

discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            scope='discriminator_model')
generator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            scope='generator_model')
print ("Printing variables for upadte")
total_count = 0
for idx, op in enumerate(generator_params):
  shape = op.get_shape()
  count = np.prod(shape)
  print ("FOR UPDATE: [%2d] %s %s = %s" % (idx, op.name, shape, count))
  total_count += int(count)
for idx, op in enumerate(discriminator_params):
  shape = op.get_shape()
  count = np.prod(shape)
  print ("FOR UPDATE: [%2d] %s %s = %s" % (idx, op.name, shape, count))
  total_count += int(count)
print ("Total %d variables to be optimized" % total_count)

# get loss gradients over multiple GPUs
xs = []
grads_gen = []
grads_dis = []
loss_gen = []
loss_gen_test = []
nll_train = []
nll_test = []
loss_dis = [] # discriminator loss
loss_dis_test = []
for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape))
    with tf.device('/gpu:%d' % i):
      # train
      
      # output of generator
      z,jacs = model(xs[i], ema=None, **model_opt)
      nll_train.append(nn.loss(z,jacs))

      # output of discriminator
      #mask = tf.round(tf.random_uniform(nn.int_shape(z)))
      
      #random_z = tf.random_normal(nn.int_shape(z)) * mask +\
      #           z * (1.0-mask)
      random_z = tf.random_normal(nn.int_shape(z))
      #random_z = tf.random_uniform(nn.int_shape(z))*0.1+0.3
      #random_z = tf.random_normal([nn.int_shape(z)[0], 4, 4, 1])
      #random_z = tf.tile(random_z,[1,nn.int_shape(z)[1]//4,nn.int_shape(z)[2]//4,nn.int_shape(z)[3]])
      
      sampled_x = inv_model(random_z)
      #distorted_x = xs[i] + tf.random_uniform(nn.int_shape(xs[i]), -0.1, 0.1)
      #distorted_x = tf.clip_by_value(distorted_x, -1.0, 1.0)
      d_logits_of_fake = discriminator_model(sampled_x, ema=None, **model_opt)
      d_logits_of_real = discriminator_model(xs[i], ema=None, **model_opt)
      #d_logits_of_fake = discriminator_model(random_z, ema=None, **model_opt)
      #d_logits_of_real = discriminator_model(z, ema=None, **model_opt)
      
      # loss of generator
      loss_gen.append(nn.loss(z, jacs, d_logits_of_fake, d_logits_of_real))

      # loss of discriminator
      loss_dis.append(dn.loss(d_logits_of_real, d_logits_of_fake))
      
      # gradients
      grads_gen.append(tf.gradients(loss_gen[i], generator_params))
      grads_dis.append(tf.gradients(loss_dis[i], discriminator_params))
      
      # test
      z,jacs = model(xs[i], ema=None, **model_opt)
      loss_gen_test.append(nn.loss(z, jacs, d_logits_of_fake, d_logits_of_real))
      nll_test.append(nn.loss(z, jacs))
      loss_dis_test.append(dn.loss(d_logits_of_real, d_logits_of_fake))

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        loss_dis[0] += loss_dis[i]
        loss_dis_test[0] += loss_dis_test[i]
        nll_train[0] += nll_train[i]
        nll_test[0] += nll_test[i]

        for j in range(len(grads_gen[0])):
            grads_gen[0][j] += grads_gen[i][j]
        for j in range(len(grads_dis[0])):
            grads_dis[0][j] += grads_dis[i][j]
    # training op
    optimizer_gen = nn.adam_updates(generator_params, grads_gen[0], lr=tf_lr, mom1=0.95, mom2=0.9995)
    optimizer_dis = nn.adam_updates(discriminator_params, grads_dis[0], lr=tf_lr, mom1=0.95, mom2=0.9995)

# convert generator loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
nll_bits_per_dim_test = nll_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# convert discriminator loss
discriminator_loss = loss_dis[0]/args.nr_gpu
discriminator_loss_test = loss_dis_test[0]/args.nr_gpu


# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
def prepro(x):
  #x[:,:,:,0] = 255.0
  #x[:,:,:,1] = np.random.uniform(0.0,0.5)*255.0
  #x[:,:,:,1] = 0.0
  #x[:,:,:,2] = 0.0

  #x = np.random.uniform(0.0,1.0,(x.shape[0], 32, 32, 3))*255.0

  # corrupt data (Tapani Raiko's dequantization)
  corruption_level = 1.0
  x = x + corruption_level * np.random.uniform(0.0,1.0,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))

  return np.cast[np.float32]((x - 128.0) / 128.0)

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


# //////////// perform training //////////////
gen_mode = True
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
            x = train_data.next(args.batch_size) # manually retrieve exactly batch_size examples
            train_data.reset() # rewind the iterator back to 0 to do one full epoch
            print('initializing the model...')
            sess.run(initializer,{x_init: prepro(x)})
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
            #util.show_all_variables()

        

        # train for one epoch
        train_losses_gen = []
        train_losses_dis = []
        last_l_dis = 0.0
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
          if gen_mode:
            # only train generator        
            l_gen, l_dis, _ = sess.run([bits_per_dim, discriminator_loss, 
                                        optimizer_gen], feed_dict)
          #if t%4==0 and last_l_dis >= 0.2:
          else:
            sys.stdout.write('dis trained ')
            l_gen, l_dis, _ = sess.run([bits_per_dim, discriminator_loss, 
                                        optimizer_dis], feed_dict)

          if l_dis > 0.4 and gen_mode:
            sys.stdout.write('[')
            gen_mode = False
          elif gen_mode == False and l_dis < 0.1:
            sys.stdout.write(']')
            gen_mode = True
          
          #l_gen, l_dis,_,_ = sess.run([bits_per_dim, discriminator_loss, 
          #                             optimizer_gen, optimizer_dis], feed_dict)
          #sys.stdout.write('discriminator trained ')          
          last_l_dis = l_dis
          if t%10==0:
            sys.stdout.write('%.4f/%.4f ' % (l_gen, l_dis))
          sys.stdout.flush()

          train_losses_gen.append(l_gen)
          train_losses_dis.append(l_dis)

        train_loss_gen = np.mean(train_losses_gen)
        train_loss_dis = np.mean(train_losses_dis)

        # compute likelihood over test split
        test_losses_gen = []
        test_nlls_gen = []
        test_losses_dis = []
        print ("Testing...")
        for x in test_data:
          xf = prepro(x)
          xfs = np.split(xf, args.nr_gpu)
          feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
          l_gen, nll_gen, l_dis  = sess.run([bits_per_dim_test, nll_bits_per_dim_test, 
                                    discriminator_loss_test], feed_dict)
          test_losses_gen.append(l_gen)
          test_nlls_gen.append(nll_gen)
          test_losses_dis.append(l_dis)
        test_loss_gen = np.mean(test_losses_gen)
        test_nll_gen = np.mean(test_nlls_gen)
        test_loss_dis = np.mean(test_losses_dis)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f discriminator train = %.4f, discriminator test = %.4f, nll test = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen, train_loss_dis, test_loss_dis, test_nll_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            print ("Generating samples...")

            # generate samples from the model
            sampled_x = sample_from_model(sess)
            img_tile = plotting.img_tile(sampled_x, aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d.png' % (args.data_set, epoch)))
            plotting.plt.close('all')
            
            # save params
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))
