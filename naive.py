import random
import time

class naive_conv:
    ins_count = 0  # Class variable to track the number of instructions executed

    def __init__(self, input_dim, kernel_dim, num_channels, num_kernels,
                 batch_size=1, stride=(1,1)):
        '''
        input_dim: tuple of (H, W) where H is the height and W is the width of the input image
        kernel_dim: tuple of (R, S) where R is the height and S is the width of the kernel
        stride(U): tuple of (dH, dW) where dH is the vertical stride and dW is the horizontal stride
        num_channels(C): number of channels in the input image
        num_kernels(M): number of kernels in the convolutional layer
        batch_size(B): number of images in the batch
        '''
        
        self.input_dim = input_dim
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.batch_size = batch_size
        
        self.H, self.W = input_dim
        self.R, self.S = kernel_dim
        self.dH, self.dW = stride
        
        self.input = [[[[self.rand() for _ in range(self.W)] for _ in range(self.H)] 
                       for _ in range(self.num_channels)] for _ in range(self.batch_size)]
        
        self.kernel = [[[[self.rand() for _ in range(self.S)] for _ in range(self.R)]
                        for _ in range(self.num_channels)] for _ in range(self.num_kernels)]
        
        self.output, self.time, self.ins_count = naive_conv.conv2d(self.input, self.kernel, self.stride, stats=True)

    def rand(self):
        '''
        Returns a random number between -1 and 1
        '''
        sign = random.choice([-1, 1])
        return sign * random.random()
    
    # Can be accessed without creating an instance of the class
    @staticmethod
    def conv2d(img, kernel, stride=(1,1), stats=False):
        '''
        img: 4D python list of shape (B, C, H, W)
        kernel: 4D python list of shape (M, C, R, S)
        '''
        naive_conv.ins_count = 0  # Reset instruction count at the start
        t0 = time.time()
        
        B, C, H, W = len(img), len(img[0]), len(img[0][0]), len(img[0][0][0])
        M, C, R, S = len(kernel), len(kernel[0]), len(kernel[0][0]), len(kernel[0][0][0])
        dH, dW = stride

        # Initialize the output tensor (B, M, H', W')
        Ho = (H - R) // dH + 1
        Wo = (W - S) // dW + 1
        out = [[[[0 for _ in range(Wo)] for _ in range(Ho)] for _ in range(M)] for _ in range(B)]
        naive_conv.ins_count += B * M * Ho * Wo  # Instructions for initializing output tensor
            
        # Perform the naive convolution
        for b in range(B):
            for m in range(M):
                for i in range(Ho):
                    for j in range(Wo):
                        for c in range(C):
                            for r in range(R):
                                for s in range(S):
                                    out[b][m][i][j] += img[b][c][i*dH + r][j*dW + s] * kernel[m][c][r][s]
                                    naive_conv.ins_count += 1  # Count multiplication and addition

        t1 = time.time()
        if stats:
            print('Naive convolution done, output shape: ',
                    len(out), len(out[0]), len(out[0][0]), len(out[0][0][0]))
            print('Time taken for naive convolution: ', t1-t0)
            print('Instructions executed(naive):', naive_conv.ins_count)
            return out, t1-t0, naive_conv.ins_count
        else:
            return out
    
    
if __name__ == '__main__':    
    # test
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

    conv = naive_conv.conv2d(img, ker, stride=(1,1))
    print('Naive method: ', conv)
    
        
    
        
    
    
    

    
    
    
    