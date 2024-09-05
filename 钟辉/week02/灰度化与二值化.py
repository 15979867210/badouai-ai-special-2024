import cv2 
import numpy as np
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png')
height,width =img.shape[:2]
img_array=np.zeros([height,width],img.dtype)  #定义一个空图片大小的数组

#灰度方法一
# img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale Image', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #灰度方法二
# img=cv2.imread('lenna.png',0)
# cv2.imshow('Grayscale Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #二值法
# img=cv2.imread('lenna.png',0)
# t,img_binary=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# cv2.imshow('Grayscale Image', img_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # plt.subplot(221)
# # plt.imshow(img_binary)

#灰度方法三, #图像灰度非线性变换
def flog(c,img):
    output=c*np.log(1+img)
    output=np.uint8(output+0.5)
    return output
# img_gray=flog(200,img)

# plt.subplot(222)
# plt.imshow(img_gray)
# cv2.imshow('Grayscale Image', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_bianry=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img_gray_change=flog(20,img)
# 原图
plt.subplot(221)
plt.imshow(img)
#灰度图
plt.subplot(222)
plt.imshow(img_gray)
#二值图
plt.subplot(223)
plt.imshow(img_binary)
cv2.imshow('Grayscale Image', img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
#灰度变化
plt.subplot(224)
plt.imshow(img_gray_change)

