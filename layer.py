import tensorflow as tf


def weight_variable(shape, name=None, mean=0.0, stddev=1.0):
    return tf.get_variable(
        "%s/weight" % name,
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev),
        dtype=tf.float32
    )

def global_avg_pool(input_tensor):
    return tf.reduce_mean(input_tensor,[1,2])

def bias_variable(shape, name=None):
    return tf.get_variable(
        "%s/bias" % name,
        shape=shape,
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32
    )

def conv2d(input_t,filter, name, stride_size=1, padding="SAME"):
    with tf.variable_scope(name):
        w_conv = weight_variable(filter)
        output_t = tf.nn.conv2d(input_t,w_conv,strides=[1,stride_size,stride_size,1],padding=padding)
        b_conv = bias_variable(filter[-1])
        output_t = tf.nn.bias_add(output_t,b_conv)
    return output_t
