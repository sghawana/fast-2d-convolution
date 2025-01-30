import random
import subprocess

from naive import naive_conv
from flatten import flatten_conv

B = 8 # Number of images in the batch
M = 32 # Number of kernels
C = 4 # Number of channels
H = 32; W = 32 # Height and Width of the image (HXW)
dH = 2; dW = 2 # Vertical and Horizontal stride (dHXdW)
R = 5; S = 5 # Height and Width of the kernel (RXS)


# Test both the flatten and naive implementations for randomly initialized inputs and kernels
print('Testing for randomly initialized inputs and kernels')

img = [[[[random.random() for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(B)]
ker = [[[[random.random() for _ in range(S)] for _ in range(R)] for _ in range(C)] for _ in range(M)]

o1, _, _ = naive_conv.conv2d(img, ker, stats=True)
o2, _, _ = flatten_conv.conv2d(img, ker, stats=True)
print('Equal outputs: ', o1 == o2)
print('\n')


# Test both the flatten and naive implementations for a Particular input and kernel
print('Testing for a particular input and kernel')
img = [[[[-8, 4, 8],
           [-2, -9, -7],
           [7, -5, 7]],  [[-7, 6, 7],
                          [-9, -3, -8],
                          [1, -3, 0]]]]

ker = [[[[-8,1],
        [-2,3]], [[-6, -6],
                    [2, 5]]],
        [[[-6,9],
        [-7,5]], [[-4, 7],
                    [-1, 0]]],
        [[[-2,-3],
        [-2,1]], [[-3, -9],
                    [1, 0]]]]

conv, _, _ = naive_conv.conv2d(img, ker, stats=True)
print('Naive method: ', conv)

conv2, _, _ = flatten_conv.conv2d(img, ker, stats=True)
print('Flatten Method: ', conv2)

print('Equal outputs: ', conv == conv2)






