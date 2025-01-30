import random 
import time

class flatten_conv():
    def __init__(self, input_dim, kernel_dim, num_channels, num_kernels,
                 batch_size = 1, stride = (1,1)):
        
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
        
        self.ins_count = None
        self.output, self.time = flatten_conv.conv2d(input = self.input, kernel= self.kernel,
                                  stride = self.stride, stats=True)
        
    def rand(self):
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
        
        B, C, H, W = len(img), len(img[0]), len(img[0][0]), len(img[0][0][0])
        M, C, R, S = len(kernel), len(kernel[0]), len(kernel[0][0]), len(kernel[0][0][0])
        dH, dW = stride
        
        #reshape the  kernel MxCxRxS to M X (C*R*S)
        temp_kernel = [[kernel[m][c][r][s] for c in range(C) for r in range(R) for s in range(S)]
                  for m in range(M)]
        
        #reshape the input image BxCxHxW to (C*R*S) x (B*(H-R+1)*(W-S+1))
        temp_img = [[img[b][c][h+i][w+j] for b in range(B) for h in range(H-R+1) for w in range(W-S+1)] 
                    for c in range(C) for i in range(R) for j in range(S)]
        
        #2d matrix product of kernel and img
        # M x (B*(H-R+1)*(W-S+1))
        temp_out = [[sum(temp_kernel[m][k] * temp_img[k][j] for k in range(C*R*S)) for j in range(B*(H-R+1)*(W-S+1))]
                    for m in range(M)]
        
        # Initialize the final output tensor (B, M, H', W')
        Ho = (H - R) // dH + 1
        Wo = (W - S) // dW + 1
        out = [[[[0 for _ in range(Wo)] for _ in range(Ho)] for _ in range(M)] for _ in range(B)]

        # Reshape temp_out (M x (B * Ho * Wo)) to (B x M x Ho x Wo)
        for m in range(M): 
            for b in range(B): 
                for h in range(Ho):  
                    for w in range(Wo): 
                        out[b][m][h][w] = temp_out[m][b * Ho * Wo + h * Wo + w]
                        
        return out
        
            
if __name__ == '__main__':
    
    # Test Kernel Transformation
    # 2X2X3X3 kernel
    F =  [[ [[1,2,3],
           [4,5,6],
           [7,8,9]], [[10,11,12],
                     [13,14,15],
                      [16,17,18]] ],
          
           [ [[19,20,21],
           [22,23,24],
           [25,26,27]], [[28,29,30],
                     [31,34,33],
                      [34,35,36]] ]
          ]
    
    Fnew = [[F[m][c][r][s] for c in range(2) for r in range(3) for s in range(3)]
                  for m in range(2)]
    print(Fnew, '\n')
    # should print [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #              [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 33, 34, 35, 36]]
    
    
    # Test Image Transformation
    # 2X2X3X3 image
    
    I =  [[ [[1,2,3],
           [4,5,6],
           [7,8,9]], [[10,11,12],
                     [13,14,15],
                      [16,17,18]] ],
          
           [ [[19,20,21],
           [22,23,24],
           [25,26,27]], [[28,29,30],
                     [31,34,33],
                      [34,35,36]] ]
          ]
    
    Inew = [[I[b][c][h+i][w+j] for b in range(2) for h in range(2) for w in range(2)] 
                for c in range(2) for i in range(2) for j in range(2)]
    print(Inew, '\n')
    #should print [[1, 2, 4, 5, 19, 20, 22, 23],
                # [2, 3, 5, 6, 20, 21, 23, 24],
                # [4, 5, 7, 8, 22, 23, 25, 26],
                # [5, 6, 8, 9, 23, 24, 26, 27],
                # [10, 11, 13, 14, 28, 29, 31, 34],
                # [11, 12, 14, 15, 29, 30, 34, 33],
                # [13, 14, 16, 17, 31, 34, 34, 35], 
                # [14, 15, 17, 18, 34, 33, 35, 36]] 
                
                
    # test the matrix product formula
    A = [[1, 2, 3], [4, 5, 6]] # 2x3 matrix
    B = [[1, 2], [3, 4], [5, 6]] # 3x2 matrix
    C = [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(2)]
                    for i in range(2)]
    print(C, '\n') # should print [[22, 28],
    #                        [49, 64]]
    
    
    # test the convolution
    img = [[[[1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16]]]]

    ker = [[[[1,2],
            [3,4]]]]

    conv = flatten_conv.conv2d(img, ker)
    print('Flatten Method: ', conv)
    
    
    
    