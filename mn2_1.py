# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

with tf.device('/cpu:0'):
  iteration = tf.Variable(tf.constant(0.0), trainable=False)
  
with tf.device('/gpu:0'):
  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])
  
  x_image = tf.reshape(x, [-1,28,28,1])
  
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

  regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

  cross_entropy += 5e-4 * regularizers

  learning_rate = tf.train.exponential_decay(0.01, iteration * 16, 5000, 0.95, staircase=True)

  train_step = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=False).minimize(cross_entropy,global_step=iteration)

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 모든 variable을 저장하는 saver생성. 
saver = tf.train.Saver(tf.all_variables())

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = mnist.train.next_batch(16)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    # 100 iteration마다 log 폴더에 variable 저장. 
    saver.save(sess, './log/model.ckpt', global_step=iteration)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# variable 저장
saver.save(sess, './log/model.ckpt', global_step=iteration)
# Total Iteration print
print("Total Iteration is %d"%(sess.run(iteration)))

test_accuracy=0
for i in range(1000):
  test_batch = mnist.test.next_batch(10)
  test_accuracy += accuracy.eval(feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
  
print("test accuracy %g"%(test_accuracy/1000))
