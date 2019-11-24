import tensorflow as tf
from tensorflow.contrib import slim


class Lenet(object):
    def __init__(self,inputs, name,scope='lenet', training_flag=True, reuse=False):
        self.scope=scope
        self.name=name
        self.inputs=inputs
        if inputs.get_shape()[3] == 3:
            self.inputs = tf.image.rgb_to_grayscale(self.inputs)
        self.training_flag=training_flag
        self.is_training=True
        self.reuse=reuse
        self.create()

    def create(self,is_training=False):

        with tf.variable_scope(self.scope, reuse=self.reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
                    net=self.inputs
                    net = slim.conv2d(net, 64, 5, scope='conv1')
                    self.conv1=net
                    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
                    self.pool1 = net
                    net = slim.conv2d(net,128, 5, scope='conv2')
                    self.conv2= net
                    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
                    self.pool2= net
                    self.cmp=net
                    net = tf.contrib.layers.flatten(net)
                    self.att_flat=net
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
                    self.fc3= net
                    net = slim.dropout(net,0.5, is_training=self.training_flag)
                    # net = slim.fully_connected(net,64, activation_fn=tf.tanh,scope='fc4')
                    net = slim.fully_connected(net,64, activation_fn=tf.nn.relu,scope='fc4')
                    self.fc4 = net
                    net = slim.fully_connected(net,10, activation_fn=None, scope='fc5')
                    self.fc5 = net
        with tf.variable_scope(self.name):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
                    net=slim.fully_connected(net,1024,activation_fn=tf.nn.relu,scope='fc6')
                    net=tf.reshape(slim.fully_connected(net,8*8*128,activation_fn=tf.nn.relu),[-1,8,8,128])
                    self.dconv1=slim.conv2d_transpose(net,64,5,stride=2)
                    self.dconv2=slim.conv2d_transpose(self.dconv1,32,5,stride=2)
                    self.re=slim.conv2d_transpose(self.dconv2,1,5,stride=1,padding='same')


                    self.softmax_output=slim.softmax( self.fc5,scope='prediction')

