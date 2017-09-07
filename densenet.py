import numpy as np
import tensorflow as tf

from layer import *


def bottleneck_layer(input_tensor,in_channels,out_channels,name,keep_prob,is_training=True):
    with tf.variable_scope(name):
        # FIX IT : temporary batch_norm with contrib
        output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training, updates_collections=None)
        output_t = tf.nn.relu(output_t)
        output_t = conv2d(output_t,[1,1,in_channels,out_channels],name="1x1conv")
        output_t = tf.nn.dropout(output_t,keep_prob)

        # FIX IT : temporary batch_norm with contrib
        output_t = tf.contrib.layers.batch_norm(output_t, scale=True, is_training=is_training, updates_collections=None)
        output_t = tf.nn.relu(output_t)
        output_t = conv2d(output_t,[3,3,in_channels,out_channels],name="3x3conv")
        output_t = tf.nn.dropout(output_t,keep_prob)
    return output_t

def transition_layer(input_tensor,in_channels,out_channels,name,keep_prob,is_training=True):
    with tf.variable_scope()

def classification_layer():
    pass

def dense_block(input_tensor,l,k,keep_prob,layer_name):
    with tf.name_scope(layer_name):
        output_t = input_tensor
        in_channels = input_tensor.get_shape()[-1]
        for i in range(0,l):
            temp = bottleneck_layer(output_t,in_channels=in_channels,out_channels=k,name="bottleN_%d" % l,keep_prob=keep_prob)
            output_t = tf.concat([output_t,temp],axis=3)
            in_channels += k
    return output_t


def build_network(input_tensor, growth_rate, name="densenet"):
    input_channel = input_tensor.get_shape()[3]

    output = tf.nn.conv2d(input_tensor,[7,7,input_channel,16],strides=2,padding="SAME",name="conv0")
    output = tf.nn.max_pool(output,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="max_pool0")





