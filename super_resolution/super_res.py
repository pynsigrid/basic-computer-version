# USAGE
# python super_res_video.py --model models/ESPCN_x4.pb

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
from cv2 import dnn_superres
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to super resolution model")
ap.add_argument("-i", "--image", required=True,
	help="path to image")
args = vars(ap.parse_args())

# extract the model name and model scale from the file path
modelName = args["model"].split("/")[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print("[INFO] loading super resolution model: {}".format(
	args["model"]))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

time.sleep(2.0)

image = cv2.imread(args["image"],1)
print(f'shape = {image.shape}')
t1 = time.time()
upscaled = sr.upsample(image)
t2 = time.time()
bicubic = cv2.resize(image,
    (upscaled.shape[1], upscaled.shape[0]),
    interpolation=cv2.INTER_CUBIC)
t3 = time.time()
# show the original frame, bicubic interpolation frame, and super
# resolution frame
print(f'the time of SR: {round(t2-t1, 2)}, bicubic: {round(t3-t2, 2)}')
cv2.imshow("Original", image)
cv2.imshow("Bicubic", bicubic)
cv2.imshow("Super Resolution", upscaled)
cv2.imwrite("test_SR.png", upscaled)
cv2.waitKey(0)