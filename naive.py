import random
import time
import subprocess

class naive_conv():
    def __init__(self, input_dim, kernel_dim, stride, num_channels, num_kernels, batch_size):
        
        '''
        input_dim: tuple of (H, W) where H is the height and W is the width of the input image
        kernel_dim: tuple of (R, S) where R is the height and S is the width of the kernel
        stride(U): tuple of (dH, dW) where dH is the vertical stride and dW is the horizontal stride
        num_channels(N): number of channels in the input image
        num_kernels(M): number of kernels in the convolutional layer
        batch_size: number of images in the batch
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
        
        self.output = self.conv2d(self.input, self.kernel, self.stride)
        
    def rand():
        '''
        Returns a random number between -1 and 1
        '''
        sign = random.choice([-1, 1])
        return sign * random.random()
    
    #can be accesses without creating an instance of the class
    @staticmethod
    def conv2d(img, kernel, stride=(1,1), stats=False):
        '''
        img: 4d python list of shape (B, C, H, W)
        kernel: 4d python list of shape (M, C, R, S)
        '''
        t0 = time.time()
        
        B, C, H, W = len(img), len(img[0]), len(img[0][0]), len(img[0][0][0])
        M, C, R, S = len(kernel), len(kernel[0]), len(kernel[0][0]), len(kernel[0][0][0])
        dH, dW = stride
        
        # Initialize the output tensor (B, M, H', W')
        Ho = (H - R) // dH + 1
        Wo = (W - S) // dW + 1
        out = [[[[0 for _ in range(Wo)] for _ in range(Ho)] for _ in range(M)] for _ in range(B)]
            
        # Perform the naive convolution
        for b in range(B):
            for m in range(M):
                for i in range(Ho):
                    for j in range(Wo):
                        for c in range(C):
                            for r in range(R):
                                for s in range(S):
                                    out[b][m][i][j] += img[b][c][i*dH + r][j*dW + s] * kernel[m][c][r][s]
                                    
        t1 = time.time()
        if stats:
            print('Convolution done, output shape: ',
                    len(out), len(out[0]), len(out[0][0]), len(out[0][0][0]))
            print('Time taken for convolution: ', t1-t0)
            return out, t1-t0
        else:
            return out
        
    
        
    
    
    

    
    
    
    