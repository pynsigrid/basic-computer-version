'''
    Yining Pan : 10th Mar 2021
'''

import math
import numpy as np
import cv2
import os
import numpy as np
from .sys import eprint
# from PIL import Image
# from scipy.signal import convolve2d


"""
PSNR
"""
def mse(img1, img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2)
    return mse

def psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)


"""
SSIM
"""
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
#     2D gaussian mask - should give the same result as MATLAB's
#     fspecial('gaussian',[shape],[sigma])

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


"""
crop large image
Yining Pan: 10th Mar 2021
"""
def crop_large_image(src,cols,rows):
    origname, ext = os.path.splitext(src)
    orig=cv2.imread(src,1)
    sum_rows=orig.shape[0]   
    sum_cols=orig.shape[1]    
    w, h = sum_rows, sum_cols
    tmp = "/home/span/super_res/superresolution/tempimg/"
    if not os.path.exists(tmp):
        os.makedirs(tmp)
        eprint(f'tmp in {tmp}')

    num_cols = int(sum_cols/cols)
    num_rows = int(sum_rows/rows)
    # if less then rows/cols, keep the rest
    for i in range(num_cols+1):
        sum_rows = w
        for j in range(num_rows+1):
            if sum_rows > 0 and sum_cols > 0:
                if sum_rows < rows and sum_cols < cols:
                    croped = orig[j*rows:w+1, i*cols:h+1, :]
                elif sum_rows < rows and sum_cols >= cols:
                    croped = orig[j*rows:w+1, i*cols:(i+1)*cols,:]
                elif sum_cols < cols and sum_rows >= rows:
                    croped = orig[j*rows:(j+1)*rows,  i*cols:h+1, :]
                else:
                    croped = orig[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:]
                
                cv2.imwrite(tmp+'temp_'+str(j)+'_'+str(i)+ext, croped)
                sum_rows -= rows
        sum_cols -= cols    
        
    return tmp, num_cols, num_rows


"""
merge images from certain path
Yining Pan: 10th Mar 2021
"""
def merge_large_picture(merge_path,sum_cols,sum_rows,cols,rows):
    filename=file_name(merge_path)
    _, ext = os.path.splitext(filename[0])
    num_cols = int(sum_cols/cols)
    num_rows = int(sum_rows/rows)
    channels = cv2.imread(filename[0],1).shape[-1]
    dst = np.zeros((cols*(num_cols+1),rows*(num_rows+1),channels),np.uint8)
    for i in range(len(filename)):
        img=cv2.imread(filename[i],1)
        roi = cv2.copyMakeBorder(img, 0,rows-img.shape[0],0,cols-img.shape[1], cv2.BORDER_CONSTANT)        
        cols_th=int(filename[i].split("_")[-1].split('.')[0])
        rows_th=int(filename[i].split("_")[-2])
        dst[rows_th*rows:(rows_th+1)*rows,cols_th*cols:(cols_th+1)*cols,:]=roi
        os.remove(filename[i])
    os.rmdir(merge_path)
    dst = dst[:sum_cols+1, :sum_rows+1, :]
    return dst


"""
pack files into list
Yining Pan: 10th Mar 2021
"""
def file_name(root_path):
    filename=[]
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1] in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                filename.append(os.path.join(root,file))
    return filename


