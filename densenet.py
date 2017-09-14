import numpy as np
import tensorflow as tf

from layer import *


class DenseNet(object):
    def __init__(self, batch_size, num_classes, image_info, depth, growth_rate, total_blocks, keep_prob):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.growth_rate = growth_rate
        self.nesterov_momentum = 0.9
        self.weight_decay = 0.0001
        self.depth = depth
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - total_blocks - 1) // total_blocks

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')

        self.image_info = image_info

        self.image_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, self.image_info.height, self.image_info.width, self.image_info.channel],
            name="image_placeholder")

        self.target_placeholder = tf.placeholder(dtype=tf.int64,
                                                 shape=[self.batch_size,],
                                                 name="target_placeholder")

        logits = self.build_densenet(self.image_placeholder)
        self.prediction = tf.nn.softmax(logits)

        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target_placeholder))
        self.cross_entropy = cross_entropy

        self.loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'weight' in var])

        tf.summary.scalar("loss/classification",self.loss)

        optimizer = tf.train.MomentumOptimizer(
            self.lr_placeholder, self.nesterov_momentum, use_nesterov=True
        )
        self.train_op = optimizer.minimize(cross_entropy + self.loss * self.weight_decay, global_step=self.global_step)

        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), self.target_placeholder)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def bottleneck_composite_layer(self, input_tensor, in_channels, out_channels, name, is_training=True):
        with tf.variable_scope(name):
            # FIX IT : temporary batch_norm with contrib
            output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True,
                                                    is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            output_t = conv2d(output_t, [1, 1, in_channels, 4 * out_channels], name="1x1conv")
            output_t = tf.nn.dropout(output_t, self.keep_prob)
            # FIX IT : temporary batch_norm with contrib
            output_t = tf.contrib.layers.batch_norm(output_t, scale=True, is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            output_t = conv2d(output_t, [3, 3, 4 * out_channels, out_channels], name="3x3conv")
            output_t = tf.nn.dropout(output_t, self.keep_prob)
        return output_t

    def transition_layer(self, input_tensor, in_channels, name, c_rate, is_training=True):
        with tf.variable_scope(name):
            output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            output_t = conv2d(output_t, [1, 1, in_channels, int(int(in_channels) * c_rate)], name="1x1conv")
            output_t = tf.nn.dropout(output_t, self.keep_prob)
            output_t = tf.nn.avg_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID",
                                      name="avg_pool")
        return output_t

    def classification_layer(self, input_tensor, name, keep_prob, is_training=True):
        with tf.variable_scope(name):
            output_t = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training,
                                                    updates_collections=None)
            output_t = tf.nn.relu(output_t)
            last_pool_k = int(output_t.get_shape()[-2])
            output_t = tf.nn.avg_pool(output_t, ksize=[1, last_pool_k, last_pool_k, 1],strides=[1,last_pool_k,last_pool_k,1],padding="VALID")
            output_t = tf.reshape(output_t, shape=[-1, int(output_t.get_shape()[-1])])
            w_fc = weight_variable([int(output_t.get_shape()[-1]),self.num_classes], name="fc")
            b_fc = bias_variable(self.num_classes, name="fc")
            logits = tf.add(tf.matmul(output_t, w_fc), b_fc)
        return logits

    def dense_block(self, input_tensor, l, k, name, is_training=True):
        with tf.variable_scope(name):
            output_t = input_tensor
            in_channels = input_tensor.get_shape()[-1]
            for i in range(l):
                temp = self.bottleneck_composite_layer(output_t, in_channels=in_channels, out_channels=k,
                                                       name="bottle_composite_%d" % i,
                                                       is_training=is_training)
                output_t = tf.concat([output_t, temp], axis=3)
                in_channels += k
        return output_t, in_channels

    def build_densenet(self, input_tensor, is_training=True, name="densenet"):
        in_channels = input_tensor.get_shape()[3]

        output_t = conv2d(input_tensor, [7, 7, in_channels, self.growth_rate * 2], stride_size=2, padding="SAME", name="conv0")
        output_t = tf.nn.max_pool(output_t, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="max_pool0")

        for i in range(self.total_blocks):
            output_t, out_channels = self.dense_block(output_t, l=self.layers_per_block, k=self.growth_rate, name="denseB_%d" % i+1,
                                                      is_training=is_training)
            if i != self.total_blocks - 1:
                output_t = self.transition_layer(output_t, in_channels=out_channels, c_rate=0.5, name="transL_%d" % i+1,
                                             is_training=is_training)

        logits = self.classification_layer(output_t, name="classL", keep_prob=self.keep_prob, is_training=is_training)

        return logits

#    def build_loss(self, output_t, target):
