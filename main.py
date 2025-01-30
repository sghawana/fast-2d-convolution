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


def count_instructions(func, *args):
    """
    Counts the number of instructions used by a function using
    """
    # Save the function to a temporary file
    with open('temp_func.py', 'w') as f:
        f.write('import time\n')
        f.write('import random\n')
        f.write('from naive import naive_conv\n')
        f.write('import __main__\n')
        f.write('__main__.naive_conv = naive_conv\n')
        f.write(f'naive_conv.{func.__name__}(*{args})\n')

    # Run the function with perf
    result = subprocess.run(['perf', 'stat', '-e', 'instructions', 'python3', 'temp_func.py'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Extract the number of instructions from the perf output
    for line in result.stderr.split('\n'):
        if 'instructions' in line:
            return int(line.split()[0].replace(',', ''))
    
    return None

# Test both the flatten and naive implementations for randomly initialized inputs and kernels







# Test both the flatten and naive implementations for a Particular input and kernel
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

conv = naive_conv.conv2d(img, ker)
print('Naive method: ', conv)

conv2 = flatten_conv.conv2d(img, ker)
print('Flatten Method: ', conv2)

print('Equal outputs: ', conv == conv2)






