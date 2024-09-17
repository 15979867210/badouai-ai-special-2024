import cv2
import numpy as np
from matplotlib import pyplot as plt

#最邻近插值
def nearest(d_h,d_w,img):
    s_h,s_w,channel=img.shape
#     建立缩放图布,类型是np.uint8
    img_change=np.zeros((d_h,d_w,channel),np.uint8)
#     计算缩放比例
    sh=d_h/s_h
    sw=d_w/s_w
#     找到原图对应的（x，y），循环赋值
    for i in range(d_h):
        for j in range(d_w):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            img_change[i,j]=img[x,y]
    return img_change

#双线性插值
def bilinear(d_h,d_w,img):
    s_h,s_w,channel=img.shape
#     判断是否与原图一致
    if d_h==s_h and d_w==s_w:
        return img
    img_change=np.zeros((d_h,d_w,channel),np.uint8)
    sh, sw=d_h/s_h, d_w/s_w
    for k in range(channel):
        for i in range(d_h):
            for j in range(d_w):
#                 计算原点（x，y）
                x=(i+0.5)/sw-0.5
                y=(j+0.5)/sh-0.5
                #对数据进行处理,定义x0，x1，y0,y1
                x0=int(np.floor(x))
                x1=min(x0+1,s_w-1)
                y0=int(np.floor(y))
                y1=min(y0+1,s_h-1)
                
                #带入公式
                temp0=(x1-x)*img[x0,y0,k]+(x-x0)*img[x1,y0,k]
                temp1=(x1-x)*img[x0,y1,k]+(x-x0)*img[x1,y1,k]
                
                img_change[i,j,k]=(y1-y)*temp0+(y-y0)*temp1
                
                
    return img_change
  
img=cv2.imread("lenna.png")
change1=nearest(800,800,img)
change2=bilinear(800,800,img)
cv2.imshow("nearest_change",change1)
cv2.imshow("bilinear_change",change2)
cv2.imshow("raw_img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#直方图与均衡化

img=cv2.imread("lenna.png",1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("raw_img",gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#灰度直方图
plt.subplot(331)
# plt.figure()  #新建一个图层
plt.hist(img.ravel(),256)  #将二维数组压平成一维数组
plt.subplot(332)
plt.hist(gray.ravel(),256)  #将二维数组压平成一维数组
plt.subplot(333)
hist=cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(hist) #将二维数组压平成一维数组

#直方图均衡化方法1
equ=cv2.equalizeHist(gray)
plt.subplot(334)
plt.hist(equ.ravel(),256)  #将二维数组压平成一维数组
# cv2.imshow("0",equ)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#直方图均衡化方法2
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) 
equ_clahe = clahe.apply(gray)
plt.subplot(335)
plt.hist(equ_clahe.ravel(),256)  #将二维数组压平成一维数组

# 图像堆叠
res = np.hstack((gray,equ,equ_clahe))
cv2.imshow("0",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show
