# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def variable_summaries(name, var):
  #Summary를 추가하는 함수 선언
  with tf.name_scope('summaries'):    #아래의 variable들은 'summaries' 이름 그룹에 속함.
    with tf.device('/cpu:0'):
      mean = tf.reduce_mean(var) # name of mean : 'summaries/Mean:0'
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):    #아래의 variable들은 'summaries/stddev' 이름 그룹에 속함.
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean))) # name of stddev : 'summaries/stddev/Sqrt:0'
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

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
  # image summary 추가
  with tf.device('/cpu:0'):
    tf.image_summary('x_image', x_image)
    
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  # summary 추가
  variable_summaries('W_conv1', W_conv1)
  variable_summaries('b_conv1', b_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  keep_prob = tf.placeholder(tf.float32)
  with tf.device('/cpu:0'):
    tf.scalar_summary('dropout_keep_probability', keep_prob)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

  regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

  cross_entropy += 5e-4 * regularizers
  with tf.device('/cpu:0'):
    tf.scalar_summary('cross_entropy_loss', cross_entropy)
  learning_rate = tf.train.exponential_decay(0.01, iteration * 16, 5000, 0.95, staircase=True)
  
  train_step = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=False).minimize(cross_entropy,global_step=iteration)

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

# summary 통합.
merged = tf.merge_all_summaries()
# summary writer 생성.
train_writer = tf.train.SummaryWriter('./tensorboard/train', sess.graph)

sess.run(tf.initialize_all_variables())
for i in range(10000): #tensorboard를 monitoring하기 위해 iteration을 크게 설정. 
  batch = mnist.train.next_batch(16)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  #summary 생성 및 train. '_' 는 결과값을 받지 않을때 사용.
  summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  #run은 eval과는 다르게 계산할 variable을 인자로 받아서 사용 가능함.
  #summary 기록 
  train_writer.add_summary(summary, i)
  
  
test_accuracy=0
for i in range(1000):
  test_batch = mnist.test.next_batch(10)
  test_accuracy += accuracy.eval(feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
  
print("test accuracy %g"%(test_accuracy/1000))
