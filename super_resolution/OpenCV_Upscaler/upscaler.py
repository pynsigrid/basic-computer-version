'''
    Yining Pan : 10th Mar 2021
    Xingchen Wang : 8th Mar 2021
'''
import os
import cv2
from utils.sys import eprint
from utils.img import crop_large_image, merge_large_picture

from interface import Upscalable

class OpenCV_Upscaler(Upscalable):
    def __init__(self, scale, mpath, name):
        self.init(scale, mpath, name)

    def init(self, scale, mpath, name):
        self.scale = scale
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(mpath)
        self.sr.setModel(name, scale)
        self.cols = 100
        self.rows = 100


    def process_large_image(self, src, dst):
        orig = cv2.imread(src, 1)
        eprint(f'original shape is {orig.shape}')
        cpath, num_cols, num_rows = crop_large_image(src=src, cols=self.cols, rows=self.rows)
        for i, filename in enumerate(os.listdir(cpath)):
            img_path = cpath + filename
            croped = cv2.imread(img_path, 1)
            upscaled = self.sr.upsample(croped)
            cv2.imwrite(img_path, upscaled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            
        merged = merge_large_picture(cpath, sum_cols=self.scale*orig.shape[0], sum_rows=self.scale*orig.shape[1], 
                                     cols=self.scale*self.cols, rows=self.scale*self.rows)
        eprint(f'upscaled shape is {merged.shape}')
        cv2.imwrite(dst, merged, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    def process_normal_image(self, src, dst):
        orig = cv2.imread(src, 1)
        upscaled = self.sr.upsample(orig)
        cv2.imwrite(dst, upscaled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    def infer(self,src, dst):
        orig = cv2.imread(src, -1)
        if orig.shape[0]*orig.shape[1] <= self.cols*self.rows:
            self.process_normal_image(src, dst)
        else:
            self.process_large_image(src, dst)

class Pytorch_Upscaler(Upscalable):
    pass

class Tensorflow_Upscaler(Upscalable):
    pass

class OpenVINO_Upscaler(Upscalable):
    pass