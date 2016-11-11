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


import pixel_cnn_pp.nn as pixel_cnn_nn
import real_nvp.nn as real_nvp_nn
import pixel_cnn_pp.plotting as plotting
import pixel_cnn_pp.scopes as scopes
from pixel_cnn_pp.model import model_spec as pixel_cnn_model_spec
from real_nvp.model import model_spec as real_nvp_model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
import data.csv_data as csv_data
from skimage import io, color
import time


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
# model
parser.add_argument('--model', type=str, default='pixel_cnn', help='model name: pixel_cnn, real_nvp')
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=192, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
# optimization
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
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
model_opt = {'pixel_cnn':{ 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'dropout_p': 0.0 },
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
x_bias = tf.placeholder(tf.float32, shape=(args.sample_batch_size, ) + obs_shape)
if args.model == 'pixel_cnn':
  gen_par = model(x_sample, ema=None, **model_opt)
  new_x_gen = nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix)
  new_x_gen_biased = nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix, 
                                                             bias = x_bias)
else:
  new_x_gen = model(x_sample, ema=None, **model_opt)
  #new_x_gen = nn.sample_from_gaussian(new_x_gen)  
def sample_from_model(sess):
    if args.model == 'pixel_cnn':
      x_gen = np.zeros((args.sample_batch_size,) + obs_shape, dtype=np.float32)
      #assert len(obs_shape) == 3, 'assumed right now'
      for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
          new_x_gen_np = sess.run(new_x_gen, {x_sample: x_gen})
          x_gen[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:].copy()
      return x_gen
    else:
      x = np.linspace(0.,2.,num=args.sample_batch_size)
      #print (x.shape)
      xx = np.tile(x, [2, 1])
      xx = np.transpose(xx)
      #print (xx.shape)
      new_x_gen_np = sess.run(new_x_gen, {x_sample:xx})
      #print (x.shape)
      #print (new_x_gen_np.shape)
      #print (new_x_gen_np)
      #print (x)
      M = np.concatenate([np.reshape(x, [-1, 1]), np.reshape(new_x_gen_np, [-1, 1])], axis=1)
      #print (M)
      return M

# get loss gradients over multiple GPUs
xs = []
loss_gen_test = []
loss_gen_test_no_sum = []
for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape))
    with tf.device('/gpu:%d' % i):
        # test
        if args.model == 'pixel_cnn':
          gen_par = model(xs[i], ema=None, **model_opt)
          loss_gen_test.append(nn.loss(xs[i], gen_par,adjust_for_variance=True))
          loss_gen_test_no_sum.append(nn.loss(xs[i], gen_par, sum_all=False))
        else:
          gen_par = model(xs[i], ema=None, **model_opt)
          loss_gen_test.append(nn.loss(xs[i], gen_par))

# convert loss to bits/dim
bits_per_dim_test = loss_gen_test[0]/(np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test_no_sum = loss_gen_test_no_sum[0]/(np.log(2.)*np.prod(obs_shape))

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
def prepro(x):
  if args.model == 'pixel_cnn':
    return np.cast[np.float32]((x - 127.5) / 127.5)
  else:
    return x

def compute_likelihood(xf):
  print ("computing likelihood of image with mean %f" % np.mean(xf))
  xfs = np.split(xf, args.nr_gpu)
  #print ("Splitting %d images into groups of size %d" % (xf.shape[0], xfs.shape[0]))
  feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
  l = sess.run(bits_per_dim_test, feed_dict)
  return l

def predict(xf, original):
  print ("predicting image with mean %f" % np.mean(xf))
  #xfs = np.split(xf, args.nr_gpu)
  #new_x_gen = nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix)
  #feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
  #prediction = sess.run(new_x_gen, feed_dict)
  x_gen = np.zeros(xf.shape)
  for yi in range(x_gen.shape[1]):
    for xi in range(x_gen.shape[2]):                      
      new_x_gen_np = sess.run(new_x_gen_biased, {x_sample: x_gen, x_bias: original})
      x_gen[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:].copy()
  prediction = x_gen
  return prediction

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

def inpaint_images(x):
  
  # take the first image and create region to inpaint
  num_samples = 32
  num_cols = 4
  num_rows = (num_samples + num_cols-1)//num_cols
  x = np.tile(x[0,:,:,:], (num_samples, 1, 1, 1))
  original = x[0].copy()
  blur_amount = 0.6
  blurred = ndimage.filters.gaussian_filter(original, blur_amount)

  plt.subplot2grid((num_rows+1, num_cols),(num_rows,0))
  plt.axis('off')
  plt.imshow(original*0.5+0.5)
  plt.title('original')

  # let's also blur the original image
  x = ndimage.filters.gaussian_filter(x, blur_amount)  
    
  plt.subplot2grid((num_rows+1, num_cols),(num_rows,2))
  plt.axis('off')
  plt.imshow(blurred*0.5+0.5)
  plt.title('blurred')

  # predict
  hole_size = 8
  middle_h = x.shape[1]//2
  middle_w = x.shape[2]//2
  hole_start_y = middle_h - hole_size//2
  hole_end_y = middle_h + hole_size//2
  hole_start_x = middle_w - hole_size//2
  hole_end_x = middle_w + hole_size//2
  x[:,hole_start_y: hole_end_y + 1,
    hole_start_x:hole_end_x + 1,:] = 1.0

  plt.subplot2grid((num_rows+1, num_cols),(num_rows,1))
  plt.axis('off')
  plt.imshow(x[0]*0.5+0.5)
  plt.title('inpainting area')


  for yi in range(hole_start_y, hole_end_y+1):
    for xi in range(hole_start_x, hole_end_x+1):
      new_x_gen_np = sess.run(new_x_gen, {x_sample: x})
      x[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:].copy()
      print ("inpainted (%d,%d)" % (yi,xi))
  
  # compute likelihoods
  l = sess.run(bits_per_dim_test_no_sum, {xs[0]: x})
  print ("likelihoods computed %s" % str(l))

  # penalize deviation from blur
  diff = np.abs((x-blurred)[:,hole_start_y:hole_end_y+1,hole_start_x:hole_end_x+1,:])
  diffsum = np.sum(diff,(1,2,3)) / ((hole_end_y-hole_start_y) * (hole_end_x-hole_start_x))
  #diffmax = np.amax(np.abs(diff),(1,2,3))
  penalty = 10.0
  l = l + penalty * diffsum

  # sort 
  order = np.argsort(l)

  # save images
  x = x*0.5 + 0.5
  idx = 0
  for ver in range(num_rows):
    for hor in range(num_cols):
      plt.subplot2grid((num_rows+1, num_cols),(ver,hor))
      plt.axis('off')
      plt.imshow(x[order[idx]])
      plt.title('%f' % l[order[idx]])
      idx = idx+1
  
  plt.savefig('test/inpaint_%s.png' % time.strftime("%m_%d_%H_%M_%S"), bbox_inches='tight')

# |x| is a 3D WxHxC numpy tensor
# |rect| is list of [top,left,bottom,right]
def inpaint_superpixel(x, bias, rect, num_samples):
  num_trials = 128
  x = np.tile(x, (num_trials, 1, 1, 1))
  print (rect)
  hole_start_y = rect[0]
  hole_end_y = rect[2]
  hole_start_x = rect[1]
  hole_end_x = rect[3]
  x[:,hole_start_y: hole_end_y + 1,
    hole_start_x:hole_end_x + 1,:] = 1.0
  for yi in range(hole_start_y, hole_end_y+1):
    for xi in range(hole_start_x, hole_end_x+1):
      new_x_gen_np = sess.run(new_x_gen, {x_sample: x})
      x[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:].copy()
      print ("inpainted (%d,%d)" % (yi,xi))
  
  # compute likelihoods
  l = sess.run(bits_per_dim_test_no_sum, {xs[0]: x})
  print ("likelihoods computed %s" % str(l))

  # penalize deviation from blur
  diff = np.abs((x-bias)[:,hole_start_y:hole_end_y+1,hole_start_x:hole_end_x+1,:])
  diffsum = np.sum(diff,(1,2,3)) / ((hole_end_y-hole_start_y) * (hole_end_x-hole_start_x))
  #diffmax = np.amax(np.abs(diff),(1,2,3))
  penalty = 100.0
  l = l + penalty * diffsum

  # sort 
  order = np.argsort(l)

  # take top scores
  order = order[:num_samples]
  best_results = np.take(x, order, axis=0)
  l = np.take(l, order, axis=0)

  print ("superpixel at %s painted" % str(rect))

  return best_results, l
  

  
  

def inpaint_images_by_superpixel(x):
  # take the first image and create region to inpaint
  num_samples = 32
  num_cols = 4
  num_rows = (num_samples + num_cols-1)//num_cols
  img_h = x.shape[1]
  img_w = x.shape[2]
  x = np.tile(x[0,:,:,:], (num_samples, 1, 1, 1))
  original = x[0].copy()
  blur_amount = 0.6
  blurred = ndimage.filters.gaussian_filter(original, blur_amount)

  plt.subplot2grid((num_rows+1, num_cols),(num_rows,0))
  plt.axis('off')
  plt.imshow(original*0.5+0.5)
  plt.title('original')

  # let's also blur the original image
  #x = ndimage.filters.gaussian_filter(x, blur_amount)  
    
  plt.subplot2grid((num_rows+1, num_cols),(num_rows,1))
  plt.axis('off')
  plt.imshow(x[0]*0.5+0.5)
  plt.title('inpainting area')

  plt.subplot2grid((num_rows+1, num_cols),(num_rows,2))
  plt.axis('off')
  plt.imshow(blurred*0.5+0.5)
  plt.title('blurred')

  # inpaint superpixel-by-superpixel
  sp_w = 4 # width of superpixel
  sp_h = 4 # height of superpixel
  assert img_h%sp_h ==0 and img_w%sp_w == 0
  painted_row_count = 0
  for si in range(img_h//sp_h//3, img_h//sp_h):
    for sj in range(img_w//sp_w):      
      x = x[0]
      rect = [si*sp_h, sj*sp_w, (si+1)*sp_h-1, (sj+1)*sp_w-1]
      #x,l = inpaint_superpixel(x, blurred, rect, num_cols)
      x,l = inpaint_superpixel(x, original, rect, num_cols)
      
    # save images if there's row permitted
    if painted_row_count < num_rows:
      for hor in range(num_cols):
        plt.subplot2grid((num_rows+1, num_cols),(painted_row_count,hor))
        plt.axis('off')
        plt.imshow(x[hor]*0.5+0.5)
        plt.title('%f' % l[hor])
      painted_row_count += 1
      plt.savefig('test/inpaint_%s.png' % time.strftime("%m_%d_%H_%M_%S"), bbox_inches='tight')
       
  plt.subplot2grid((num_rows+1, num_cols),(num_rows,3))
  plt.axis('off')
  plt.title('final')
  plt.imshow(x[0]*0.5+0.5)

  plt.savefig('test/inpaint_%s.png' % time.strftime("%m_%d_%H_%M_%S"), bbox_inches='tight')
  



def create_montage_image(images, height, width, n_row, n_col, n_channel):
  images = images.reshape((n_row, n_col, height, width, n_channel))
  images = images.transpose(0, 2, 1, 3, 4)
  images = images.reshape((height * n_row, width * n_col, n_channel))
  if n_channel == 1:
    images = np.tile(images, (1,1,3))
  return images


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
      #istortions['tint_a'] = a*4
      #istortions['tint_b'] = b*4
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

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')


def save_images(images, height, width, n_row, n_col, n_channel,
      cmin=0.0, cmax=1.0, directory="./", prefix="sample"):
  
  images = create_montage_image(images, height, width, n_row, n_col, n_channel)

  filename = '%s_%s.png' % (prefix, get_timestamp())  
  scipy.misc.toimage(images) \
      .save(os.path.join(directory, filename))

  return images.reshape((1,height * n_row, width * n_col, 3))


def transform_images(x):
  num_iterations = 2
  num_images = x.shape[0]
  num_images_to_show = 4
  height = x.shape[1]
  width = x.shape[2]
  results = np.ones((num_images_to_show*num_iterations, height, width, 3))
  original = x[:num_images_to_show,:,:,:]
  original = ndimage.filters.gaussian_filter(original, 0.6)
  transformed = original
  for i in range(num_iterations):
    np.copyto(results[i*num_images_to_show:(i+1)*num_images_to_show,:,:,:], 
              transformed)
    #predicted_dist = compute_prediction((transformed-0.5)*2.0)
    #log_prediction = np.log(predicted_dist)
    #conditioned_prediction = log_prediction #-l2*0.002 
    #print (conditioned_prediction.shape)
    #next_candidate = np.argmax(conditioned_prediction, 4, None)
    if i >= num_iterations-1:
      break
    next_candidate = predict(transformed, original)
    transformed = next_candidate
  save_images(results*0.5+0.5, height, width, num_iterations, num_images_to_show, 3, 
                directory="test", prefix="transformed")
  print ("transformed iamges saved")
      

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
with tf.Session() as sess:
    for epoch in range(args.max_epochs):
        begin = time.time()

        # init
        if epoch == 0:
            x = train_data.next(args.init_batch_size) # manually retrieve exactly init_batch_size examples
            train_data.reset() # rewind the iterator back to 0 to do one full epoch
            print('initializing the model...')
            sess.run(initializer,{x_init: prepro(x)})
            ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
            print('restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        # compute likelihood
        test_losses = []
        skip_test = False
        if not skip_test:
          print ("Testing...")
          idx = 0
          for x in test_data:
            if idx >= 1:
              break
            ++idx
            xf = prepro(x)
            #transform_images(xf)
            test_likelihoods(xf)
            #inpaint_images_by_superpixel(xf)
            #inpaint_images(xf)
          #test_losses.append(l)
          test_loss_gen = np.mean(test_losses)
          test_bpd.append(test_loss_gen)
          print ("test_loss_gen=%f" % test_loss_gen)

        

        print ("Generating samples...")

        # generate samples from the model
        if args.model == 'pixel_cnn':
          sample_x = sample_from_model(sess)
          img_tile = plotting.img_tile(sample_x, aspect_ratio=1.0, border_color=1.0, stretch=True)
          img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
          plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d.png' % (args.data_set, epoch)))
          plotting.plt.close('all')
        else:
          sample_x = sample_from_model(sess)
          plt.clf()
          plt.scatter(sample_x[:,0], sample_x[:,1])
          plt.savefig("real_nvp_samples.png")

          # save params
          saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
          np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))
