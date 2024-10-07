import numpy as np
import cv2
from numpy import shape 
import random
from skimage import util

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    
    #设置噪点百分比
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    #对设置的噪点进行随机生成
    for i in range(NoiseNum):
        #随机生成x，y坐标
        randx=random.randint(0,src.shape[0]-1)
        randy=random.randint(0,src.shape[1]-1)
        #赋值高斯噪声
        NoiseImg[randx,randy]=NoiseImg[randx,randy]+random.gauss(means,sigma)
        
        if NoiseImg[randx,randy]<0:
            NoiseImg[randx,randy]=0
        elif NoiseImg[randx,randy]>255:
            NoiseImg[randx,randy]=255
    return NoiseImg


def jiaoyan(src,percetage):
    NoiseImg=src
    Noisenum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(Noisenum):
        randx=random.randint(0,src.shape[0]-1)
        randy=random.randint(0,src.shape[1]-1)
        
        if random.random()<=0.5:
            NoiseImg[randx,randy]=0
        else :
            NoiseImg[randx,randy]=255
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=GaussianNoise(img.copy(),2,4,0.5)
img2=jiaoyan(img.copy(),0.5)

#cv2代码util.random_noise直接调用
img11=util.random_noise(img.copy(),mode='gaussian')
img22=util.random_noise(img.copy(),mode='s&p',amount=0.5)
img33=util.random_noise(img.copy(),mode='poisson')
img44=util.random_noise(img.copy(),mode='speckle')

cv2.imshow('raw',img)
cv2.imshow('Gaussian', img1)
cv2.imshow('Gaussian1', img11)
cv2.imshow('jiaoyan',img2)
cv2.imshow('jiaoyan1',img22)
cv2.imshow('bosong',img33)
cv2.imshow('junyun',img44)

cv2.waitKey(0)
cv2.destroyAllWindows()
