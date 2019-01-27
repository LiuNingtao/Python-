import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

'''
index = 6
plt.imshow(X_train_orig[index])
plt.show()
print('Y(label) = '+str(np.squeeze(Y_train_orig[:, index])))
'''

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

'''
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
'''

def create_placeholders(n_H0, n_W0, n_C0, n_y):

    '''
    Creates the placeholders for the tensorflow session.
    :param n_H0:scalar, height of an input image
    :param n_W0:scalar, width of an input image
    :param n_C:scalar, number of channels of the input
    :param n_y:scalar, number of classes
    :return:
        X: placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
        Y: placeholder for the input labels, of shape [None, n_y] and dtype "float"
    '''

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


'''
X, Y = create_placeholders(64, 64, 3, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
'''


def initialize_parameters():


    '''
    initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    :return:
        :parameters: a dictionary of tensors containing W1, W2
    '''

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parametes = {'W1': W1, 'W2': W2}

    return parametes

tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print('W1 = ' + str(parameters['W1'].eval()[1, 1, 1]))
    print('W2 = ' + str(parameters['W2'].eval()[1, 1, 1]))
