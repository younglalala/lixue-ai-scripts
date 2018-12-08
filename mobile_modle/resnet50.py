import tensorflow as tf
import numpy as np


def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters    #这个单元中三个卷积的输出层
    with tf.variable_scope(block_name):
        X_shortcut = X_input   #X_shortcut为上一层的输出，一部分传入本次网络进行计算，另外复制一份给下一次网络进行计算

        # first
        W_conv1 = self.weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # second
        W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # third

        W_conv3 = self.weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        # final step
        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result




def convolutional_block(self, X_input, kernel_size, in_filter,
                        out_filters, stage, block, training, stride=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block   #网络域名
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input    #shortcut 部分单独提取出来作为下一次输入
        #first
        W_conv1 = self.weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #second
        W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #third
        W_conv3 = self.weight_variable([1,1, f2,f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        #shortcut path
        W_shortcut = self.weight_variable([1, 1, in_filter, f3])    #shortcut的权重，要保证输出的通道和shotrcut的通道相同才能相加
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        #final
        add = tf.add(x_shortcut, X)   #把输出和shortcut加起来作为下一次的输入。
        add_result = tf.nn.relu(add)

    return add_result

def deepnn(self, x_input, classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """
    x = tf.pad(x_input, tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]]), "CONSTANT")  #用0填充

    with tf.variable_scope('reference') :
        training = tf.placeholder(tf.bool, name='training')   #true or false

        #stage 1
        w_conv1 = self.weight_variable([7, 7, 3, 64])
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='VALID')
        assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

        #stage 2
        x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
        x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
        x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

        #stage 3
        x = self.convolutional_block(x, 3, 256, [128,128,512], 3, 'a', training)
        x = self.identity_block(x, 3, 512, [128,128,512], 3, 'b', training=training)
        x = self.identity_block(x, 3, 512, [128,128,512], 3, 'c', training=training)
        x = self.identity_block(x, 3, 512, [128,128,512], 3, 'd', training=training)

        #stage 4
        x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
        x = self.identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

        #stage 5
        x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
        x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
        x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

        x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')

        flatten = tf.layers.flatten(x)
        x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x, keep_prob)

        logits = tf.layers.dense(x, units=6, activation=tf.nn.softmax)

    return logits, keep_prob, training

