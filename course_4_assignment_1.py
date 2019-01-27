import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X, pad):

    '''
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.
    :param X:python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    :param pad:integer, amount of padding around each image on vertical and horizontal dimensions
    :return:
    X_pad:padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    '''

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    return X_pad
'''
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print('x.shape = ', x.shape)
print('x_pad.shape', x_pad.shape)
print('x[1, 1] = ', x[1, 1])
print('x_pad[1, 1] = ', x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
fig.show()
'''

def conv_single_step(a_slice_prev, W, b):

    '''
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.
    :param a_slice_prev:slice of input data of shape (f, f, n_C_prev)
    :param W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    :param b:Bias parameters contained in a window - matrix of shape (1, 1, 1)
    :return:
    Z:a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    '''

    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)

    return Z


'''
np.random.seed(1)
a_slice_prev= np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)
Z = conv_single_step(a_slice_prev, W, b)
print('Z=', Z)
'''

def conv_forward(A_prev, W, b, h_paramaters):

    '''
    Implements the forward propagation for a convolution function
    :param A_prev:output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param W:Weights, numpy array of shape (f, f, n_C_prev, n_C)
    :param b:Biases, numpy array of shape (1, 1, 1, n_C)
    :param h_paramaters:python dictionary containing "stride" and "pad"
    :return:
        Z: conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache: cache of values needed for the conv_backward() function
    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = h_paramaters['stride']
    pad = h_paramaters['pad']

    # 纵向卷积次数
    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    # 横向卷积次数
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    # 计算卷积区域
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[..., c], b[..., c])
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, h_paramaters)
    return Z, cache

'''
np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
h_parameters = {'pad': 2,
                'stride': 1}

Z, conv_cache = conv_forward(A_prev, W, b, h_parameters)
print("Z's mean = ", np.mean(Z))
print('cache_conv[0][1][2][3] = ', conv_cache[0][1][2][3])
'''


def pool_forward(A_prev, h_parameters, mode='max'):

    '''
    Implements the forward pass of the pooling layer
    :param A_prev: Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param h_parameters:python dictionary containing "f" and "stride"
    :param mode:the pooling mode you would like to use, defined as a string ("max" or "average")
    :return:
        A:output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache:cache used in the backward pass of the pooling layer, contains the input and hparameters
    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = h_parameters['f']
    stride = h_parameters['stride']

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 计算池化区域
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, h_parameters)

    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache


'''
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
h_parameters = {'stride': 1, 'f': 4}

A, cache = pool_forward(A_prev, h_parameters)
print('mode =  max')
print('A = ', A)
print()
print('mode = average')
print('A = ', A)
'''

