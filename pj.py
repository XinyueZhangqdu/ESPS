import cv2
import numpy as np
import os
img_dir = './'
result_dir = './'
pj_dir = './'

if not os.path.exists(pj_dir):
    os.mkdir(pj_dir)
img_name = os.listdir(img_dir)
for imgname in img_name:
    print(imgname)
    img = cv2.imread(img_dir+imgname)
    result = cv2.imread(result_dir+imgname)
    #print(img.shape,result.shape)
    img = cv2.resize(img,(512,256))
    #print(img.shape,result.shape)
    #img = cv2.resize(img,None,fx=0.5,fy=0.5)
    #result = cv2.resize(result,None,fx=0.5,fy=0.5)
    try:
        htich = np.hstack((img,result))
        cv2.imwrite(pj_dir+imgname,htich)
    except:
        print('skip')
