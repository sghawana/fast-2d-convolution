import random
import subprocess

from naive import naive_conv

B = 8 # Number of images in the batch
M = 32 # Number of kernels
C = 4 # Number of channels
H = 32; W = 32 # Height and Width of the image (HXW)
dH = 1; dW = 1 # Vertical and Horizontal stride (dHXdW)
R = 5; S = 5 # Height and Width of the kernel (RXS)


def rand():
    '''
    Returns a random number between -1 and 1
    '''
    sign = random.choice([-1, 1])
    return sign * random.random()


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

# Example usage
img = [[[[rand() for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(B)]
kernel = [[[[rand() for _ in range(S)] for _ in range(R)] for _ in range(C)] for _ in range(M)]
print(count_instructions(naive_conv.conv2d, img, kernel, (dH, dW)))






