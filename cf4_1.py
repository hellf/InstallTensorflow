# -*- coding: utf-8 -*-
import os.path
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
NUM_GPU=2
BATCH_SIZE=128

import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10

# variable�� share�ϱ� ���� get_variable()�� ���.
def weight_variable(shape, name):
  initializer = tf.truncated_normal_initializer(stddev=5e-2)
  return tf.get_variable(name, shape, initializer=initializer)

def bias_variable(shape, name):
  initializer = tf.constant_initializer(0.0)
  return tf.get_variable(name, shape, initializer=initializer)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME' )
  
cifar10.maybe_download_and_extract()
with tf.Graph().as_default(), tf.device('/cpu:0'):
  opt = tf.train.AdamOptimizer(1e-4)

  # Calculate the gradients for each model tower.
  tower_grads = []
  for i in xrange(NUM_GPU):
    with tf.device('/gpu:%d' % i):
      with tf.name_scope('%s_%d' % ('tower', i)) as scope:
        #train augmentation code�� �����ϰ� ������ tensorflow/models/image/cifar10.py�� ����.
        images, labels = cifar10.distorted_inputs()
        
        # conv1. ���ذ� ������ ª�� layer�� ���
        with tf.variable_scope('h_conv1') as scope:
          W_conv1 = weight_variable([5, 5, 3, 64], 'W_conv1')
          b_conv1 = bias_variable([64], 'b_conv1')
          h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1, name=scope.name)
        
        # pool1
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='h_pool1')
        
        # fc layer + softmax
        with tf.variable_scope('logits') as scope:
          W_fc1 = weight_variable([12 * 12 * 64, 10], 'W_fc1')
          b_fc1 = bias_variable([10], 'b_fc1')
          h_pool1_flat = tf.reshape(h_pool1, [-1, 12*12*64])
          logits=tf.nn.softmax(tf.matmul(h_pool1_flat, W_fc1) + b_fc1, name=scope.name)
        
        #cross entropy loss ����
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
        
  ################################################################################################################################        
        # ���� �̸��� variable�� ���� tower�� ����� �� �ֵ��� ����
        tf.get_variable_scope().reuse_variables()
        
        # gradient ����
        grads = opt.compute_gradients(loss)
        
        # Keep track of the gradients across all towers.
        tower_grads.append(grads)
        
        #tower_grads�� ������ ���� ���·� ����Ǿ�����
        #tower_grads=[((grad0_gpu0, var0_gpu0), ... , (gradM_gpu0, varM_gpu0))...((grad0_gpuN, var0_gpuN), ... , (gradM_gpuN, varM_gpuN))]
                
  # tower_grads�� average_grads�� �����Ͽ� grads�� ������Ʈ
  average_grads = []
  # grad = gradients, vars = variables
  for grad_and_vars in zip(*tower_grads):    #for var0~varM
    #for zip���� list(tower_grads)�� slice �Ͽ� grad_and_vars�� ������ ���� ���°� ��. (var0 ����)
    #grad_and_vars=((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads_ = []
    for g, _ in grad_and_vars:  #for gpu0~gpuN
      # g = grad0_gpu0 (gpu0 ����)
      
      # g�� dimmesion�� 1�� Ȯ��. ex) shape of g=[2 2], shape of expended_g=[1 2 2]
      # ù dimmension�� tower�� index�� �ǹ���. (g, 0):before first. (g, -1):after last
      expanded_g = tf.expand_dims(g, 0)
      # expanded_g�� �̾� �ٿ��� average�� ����
      grads_.append(expanded_g)
    # tower(gpu) dimmension���� average ����
    grad = tf.concat(0, grads_)
    grad = tf.reduce_mean(grad, 0)
    #��� tower�� variable�� �����ǹǷ� ù��° tower�� gradient�� ���ΰ���� gradient�� �ٲ�.
    v = grad_and_vars[0][1] #var0, var1, ... varM�� �ǹ���.
    grad_and_var = (grad, v) #��� grad�� ������ grad�� ��ü��. var�� ������ ����.
    average_grads.append(grad_and_var)
    
  grads = average_grads
  
  # Apply the gradients to adjust the shared variables.
  train_op = opt.apply_gradients(grads)
  ################################################################################################################################
  
  
  # Start running operations on the Graph. allow_soft_placement must be set to
  # True to build towers on GPU, as some of the ops do not have GPU
  # implementations.
  config=tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.initialize_all_variables())

  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)

  for step in xrange(1000):
    _, loss_value = sess.run([train_op, loss])

    if step % 100 == 0:
      format_str = ('step %d, loss = %.2f')
      print (format_str % (step, loss_value))


