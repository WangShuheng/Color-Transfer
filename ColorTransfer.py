import cv2
import numpy as np
import scipy.misc
from scipy.misc import imread,imresize


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
image = cv2.imread('images/test.jpg')
image = imresize(image,(IMAGE_HEIGHT,IMAGE_WIDTH))
original = cv2.imread('images/scream.jpg')
original = imresize(original,(IMAGE_HEIGHT,IMAGE_WIDTH))
height,width,channel = image.shape

def RGB2lab(image):
    r = np.zeros(image.shape,dtype=float)
    temp = np.dot([[1/np.sqrt(3),0,0],[0,1/np.sqrt(6),0],[0,0,1/np.sqrt(2)]],[[1,1,1],[1,1,-2],[1,-1,0]])
    for i in range(0,height):
        for j in range(0,width):
            lms = np.dot([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]],image[i,j,:3])
            for k in range(channel):
                if lms[k]>0:
                    lms[k]=np.log10(lms[k])
            lab = np.dot(temp,lms)
            r[i,j] = lab
    return r

def lab2RGB(image):
    r = np.zeros(image.shape,dtype=int)
    temp = np.dot([[1,1,1],[1,1,-1],[1,-2,0]],[[np.sqrt(3)/3,0,0],[0,np.sqrt(6)/6,0],[0,0,np.sqrt(2)/2]])
    for i in range(0,height):
        for j in range(0,width):
            lms = 10 ** np.dot(temp,image[i,j,:3])
            rgb = np.dot([[4.4679,-3.5873,0.1193],[-1.2186,2.3809,-0.1624],[0.0479,-0.2439,1.2045]],lms)
            r[i,j] = rgb
            for k in range(channel):
                if r[i,j,k] >= 255:
                    r[i,j,k] = 255
                elif r[i,j,k] <=0:
                    r[i,j,k] = 0
    return r

image = RGB2lab(image)
original = RGB2lab(original)

def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:,:,0])
    image_std_l = np.std(image[:,:,0])
    image_avg_a = np.mean(image[:,:,1])
    image_std_a = np.std(image[:,:,1])
    image_avg_b = np.mean(image[:,:,2])
    image_std_b = np.std(image[:,:,2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg,std)

image_avg,image_std = getavgstd(image)
original_avg,original_std = getavgstd(original)

for i in range(0,height):
    for j in range(0,width):
        for k in range(0,channel):
            t = image[i,j,k]
            t = (t-image_avg[k])*(original_std[k]/image_std[k]) + original_avg[k]
            image[i,j,k] = t
image = lab2RGB(image)
cv2.imwrite('out.jpg',image)
