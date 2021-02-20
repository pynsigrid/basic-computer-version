from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
from cv2 import dnn_superres
import os
import time
import datetime
# from skimage.measure import structural_similarity as ssim
from utils import compute_ssim, psnr

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to super resolution model")
ap.add_argument("-p", "--path", required=True,
	help="path of images")
args = vars(ap.parse_args())

# extract the model name and model scale from the file path
modelName = args["model"].split("/")[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

print("[INFO] loading super resolution model: {}".format(
	args["model"]))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)
time.sleep(2.0)

# batch process
for filename in os.listdir(args["path"]):
    print(filename)
    try:
        image = cv2.imread(args["path"] + "/" + filename, 1)
        t1 = time.time()
        upscaled = sr.upsample(image)
        t2 = time.time()
        bicubic = cv2.resize(image,
            (upscaled.shape[1], upscaled.shape[0]),
            interpolation=cv2.INTER_CUBIC)
        
        (B1, G1, R1) = cv2.split(bicubic)
        (B2, G2, R2) = cv2.split(upscaled)
        ssim1 = compute_ssim(B1, B2)
        ssim2 = compute_ssim(G1, G2)
        ssim3 = compute_ssim(R1, R2)
        mssim = (ssim1+ssim2+ssim3)/3
        mssim_res = str(round(mssim*100)) + 'E-2' if mssim < 1 else str(mssim)
        psnr1 = psnr(B1, B2)
        psnr2 = psnr(G1, G2)
        psnr3 = psnr(R1, R2)
        mpsnr = (psnr1+psnr2+psnr3)/3
        mpsnr_res = str(round(mpsnr*100)) + 'E-2' if mpsnr < 1 else str(round(mpsnr))
        out_filename = filename.split(".png")[0] + "_" + str(modelName) + "_x" + str(modelScale) + "_t" + str(round(t2-t1, 2)) + "_s" + str(mssim_res) + "_p" + str(mpsnr_res) + ".png"
        print(out_filename)
        # print(f'the ssim of {out_filename} is {mssim}')
        cv2.imwrite(args["path"] + "/result/" + str(out_filename), upscaled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        
    except:

        # print("find .DS_Store!")
        continue

