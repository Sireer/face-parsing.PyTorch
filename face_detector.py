import os
import glob
import cv2
import numpy as np

imglist = glob.glob("/mnt/lustre/wangzhibo/ffhq/images1024x1024/*.png")
imglist.sort()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


faces_list = []
for filename in imglist:
    img = cv2.imread(filename)
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_list.append(faces)
    print(faces)
    break

faces_list = np.asarray(faces_list)
np.save("./ffhq.npy", faces_list)