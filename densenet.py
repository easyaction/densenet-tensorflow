import numpy as np
import tensorflow as tf

from layer import *

class DenseNet(object):
    def __init__(self,batch_size, num_classes, keep_prob):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.keep_prob = keep_prob

    def bottleneck_layer(self,input_tensor, in_channels, out_channels, name, keep_prob, is_training=True):
        with tf.variable_scope(name):
            # FIX IT : temporary batch_norm with contrib
            output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            output_t = conv2d(output_t, [1, 1, in_channels, 4 * out_channels], name="1x1conv")
            output_t = tf.nn.dropout(output_t, keep_prob)

            # FIX IT : temporary batch_norm with contrib
            output_t = tf.contrib.layers.batch_norm(output_t, scale=True, is_training=is_training, updates_collections=None)
            output_t = tf.nn.relu(output_t)
            output_t = conv2d(output_t, [3, 3, in_channels, out_channels], name="3x3conv")
            output_t = tf.nn.dropout(output_t, keep_prob)
        return output_t

    def transition_layer(self,input_tensor, in_channels, name, keep_prob, c_rate, is_training=True):
        with tf.variable_scope(name):
            output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            output_t = conv2d(output_t, [1, 1, in_channels, int(in_channels * c_rate)], name="1x1conv")
            output_t = tf.nn.dropout(output_t, keep_prob)
            output_t = tf.nn.avg_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="avg_pool")
        return output_t

    def classification_layer(self,input_tensor,num_classes, name, keep_prob, is_training=True):
        with tf.variable_scope(name):
            output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            last_pool_k = int(output_t.get_shape()[-2])
            output_t = tf.nn.avg_pool(output_t,ksize=[1,last_pool_k,last_pool_k,1])
            output_t = tf.reshape(output_t, shape=[-1,num_classes])
            w_fc = weight_variable(output_t.get_shape().tolist(),name="fc")
            b_fc = bias_variable(num_classes,name="fc")
            logits = tf.add(tf.matmul(output_t,w_fc),b_fc)
        return logits

    def dense_block(self,input_tensor, l, k, keep_prob, name, is_training=True):
        with tf.name_scope(name):
            output_t = input_tensor
            in_channels = input_tensor.get_shape()[-1]
            for i in range(0, l):
                temp = self.bottleneck_layer(output_t, in_channels=in_channels, out_channels=k, name="bottleN_%d" % l,
                                        keep_prob=keep_prob, is_training=is_training)
                output_t = tf.concat([output_t, temp], axis=3)
                in_channels += k
        return output_t, in_channels

    def build_densenet(self,input_tensor, num_classes, growth_rate=12,is_training=True,keep_prob=self.keep_prob, name="densenet"):
        in_channels = input_tensor.get_shape()[3]

        output_t = conv2d(input_tensor, [7, 7, in_channels, 16], strides=2, padding="SAME", name="conv0")
        output_t = tf.nn.max_pool(output_t, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="max_pool0")

        output_t, out_channels = self.dense_block(output_t, l=6, k=growth_rate, keep_prob=keep_prob, name="denseB_%d" % 1, is_training=is_training)
        output_t = self.transition_layer(output_t, in_channels=out_channels, c_rate=0.5, name="transL_%d" % 1, is_training=is_training)

        output_t, out_channels = self.dense_block(output_t, l=12, k=growth_rate, keep_prob=keep_prob, name="denseB_%d" % 2, is_training=is_training)
        output_t = self.transition_layer(output_t, in_channels=out_channels, c_rate=0.5, name="transL_%d" % 2, is_training=is_training)

        output_t, out_channels = self.dense_block(output_t, l=24, k=growth_rate, keep_prob=keep_prob, name="denseB_%d" % 3, is_training=is_training)
        output_t = self.transition_layer(output_t, in_channels=out_channels, c_rate=0.5, name="transL_%d" % 3, is_training=is_training)

        output_t, out_channels = self.dense_block(output_t, l=16, k=growth_rate, keep_prob=keep_prob, name="denseB_%d" % 4, is_training=is_training)

        logits = self.classification_layer(output_t,num_classes=num_classes,name="classL",keep_prob=keep_prob,is_training=is_training)

        return logits

    def build_loss(self,output, target, learning_rate,nesterov_momentum, weight_decay ):
        pass


