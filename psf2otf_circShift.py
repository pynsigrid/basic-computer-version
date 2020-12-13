import numpy as np
import math
import cv2



def circShift(array,K):
    height,width = array.shape
    if len(array.shape)>=2 and height*width>=4:
        if type(K) ==int and abs(K)<height:
            updownA = array[:-K, :]
            mainArray = array[-K:,:]
            flip_matrix =  np.concatenate((mainArray,updownA),axis=0)
        elif type(K) ==tuple or type(K) ==list and abs(K[0])<height and abs(K[1])<width:
            updownA = array[:-K[0], :]
            mainArray = array[-K[0]:, :]
            temp = np.concatenate((mainArray, updownA), axis=0)

            leftrightA = temp[:, :-K[1]]
            tempArray = temp[:, -K[1]:]
            flip_matrix =  np.concatenate((tempArray,leftrightA),axis=1)
        else:
            print("移动的数组必须小于待移动的数组长与宽")
            flip_matrix = None
    else:
        print('传入数据错误或移动的Numpy.ndarray数组维度至少为(2，2)')
        flip_matrix = None
    return flip_matrix

def psf2otf(psf, outsize):
    if np.count_nonzero(psf) > 1:
        outheight, outwidth = outsize
        psfheight, psfwidth = psf.shape[:2]
        paddheight, paddwidth = outheight - psfheight, outwidth - psfwidth
        print('paddheight, paddwidth',paddheight, paddwidth)
        print('paddheight//2, paddwidth//2', paddheight//2, paddwidth//2)
        if paddheight==0 and paddwidth==0:
            otf = np.fft.fft2(psf)
            otf = np.real(otf),0
        else:
            otf0 = cv2.copyMakeBorder(psf, paddheight // 2, paddheight // 2,
                                     paddwidth // 2, paddwidth // 2,cv2.BORDER_CONSTANT)
            K = (-(math.floor(otf0.shape[0])//2),-(math.floor(otf0.shape[1])//2))
            otf = circShift(otf0,K)
            otfComplex = np.fft.fft2(otf)
            otf = np.real(otfComplex)
    else:
        print('该 ndrray 数组不需要转换')
        otf =  None
    return otf