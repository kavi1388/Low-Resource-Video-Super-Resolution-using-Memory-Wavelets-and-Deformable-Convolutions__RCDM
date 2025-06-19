import numpy as np
import math
import cv2
import torch

def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim_(img1, img2):
    
    img1 = img1*255.0
    img2 = img2*255.0
    
#     img1 = rgb2ycbcr(img1.astype(np.float32) / 255.) * 255.
#     img2 = rgb2ycbcr(img2.astype(np.float32) / 255.) * 255.
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def ssim_calc(x_b,y_b):
    ##x_,y_b shape b,3, 720, 1280
    _ssim = []
    for i in range(x_b.shape[0]):
        x = x_b[i]
#         print(x.shape)
        im_x = x.permute(1, 2, 0).cpu().detach().numpy()
    
        y = y_b[i]
# #         print(y.shape)
        im_y = y.permute(1, 2, 0).cpu().detach().numpy()
      
        _ssim.append(ssim_(im_x,im_y))
        return torch.tensor(sum(_ssim)/len(_ssim),dtype=torch.float)
