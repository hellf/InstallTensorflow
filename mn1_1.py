# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#0��° GPU�� visible���·� ����. �̷��� ������ ��� �ٸ� GPU�� ����Ҽ�����.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#data load
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME' )

#with tf.device() ���������� 1tab �鿩���� line�� gpu:0������ �۵��ϵ��� �ڿ��� �Ҵ��.
with tf.device('/gpu:0'):
  #placeholder�� ����ڰ� feed�� ���� �Է��ϱ����� variable�μ� run�Ǵ� eval��� ����� feeding���־�� ��.  
  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])
  #tensor�� shape�� activation�� ��� Batch x Height x Width x Channel
  #Kernel�� ��� Height x Width x Channel x Filter �� ǥ����.  
  x_image = tf.reshape(x, [-1,28,28,1])
  
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  W_fc1 = weight_variable([14 * 14 * 32, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool1, [-1, 14*14*32])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  #dropout�� keep probability(1-drop probability) �� placeholder�� ����.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  #cross entropy loss ����
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
  #adam optimizer
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  #��Ȯ�� 
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#session ����, log_device_placement=True => �ڿ��Ҵ� ǥ�� 
config = tf.ConfigProto(log_device_placement=False)
#allow_growth = True GPU �޸� �ڿ��� �ּ������� �Ҵ�
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = mnist.train.next_batch(16) #16 batch size
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0}) #image, label, keep_prob feeding
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #image, label, keep_prob feeding

test_accuracy=0
for i in range(1000):
  test_batch = mnist.test.next_batch(10)
  test_accuracy += accuracy.eval(feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
  
print("test accuracy %g"%(test_accuracy/1000))
