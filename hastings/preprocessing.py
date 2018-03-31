import cv2
from skimage.io import imsave, imread
import numpy as np

def threshold(image,t):
    '''
        Setting threhold to the image. Turn pixels below threshold black.

        Args:
            image: the input grayscale image
                   type: ndarray, shape: (dim1,dim2)
            t: the threshold
               type: INT

        Return:
            new_image: the transformed image
        '''
    _, new_image = cv2.threshold(image,t,255,cv2.THRESH_BINARY)
    return new_image

def blur(image,b):
    '''
        Smoothing the image

        Args:
            image: the input image
                   type: ndarray, shape: (dim1,dim2)
            b: size of median filter

        Return:
            the transformed image
        '''
    return cv2.medianBlur(image,b)

def gray2rgb(image):
    '''
        Transform grayscale image to RGB image

        Args:
            image: the input grayscale image
                   type: ndarray, shape: (dim1,dim2)

        Return:
            the transformed RGB image
        '''
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

def rgb2gray(image):
    '''
        Transform RGB image to grayscale image

        Args:
            image: the input RGB image
                   type: ndarray, shape: (dim1,dim2)

        Return:
            the transformed grayscale image
        '''
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def rgb2hsv(image):
    '''
        Transform BGR image to HSV image

        Args:
            image: the input RGB image
                   type: ndarray, shape: (dim1,dim2)

        Return:
            the transformed HSV image
        '''
    return cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

def hsv2rgb(image):
    '''
        Transform HSV image to RGB image

        Args:
            image: the input HSV image
                   type: ndarray, shape: (dim1,dim2)

        Return:
            the transformed RGB image
        '''
    return cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

def modifyMask(path,hash):
    '''
    Transform mask to a new mask which only contains cilia information marked
    as 1 and the cell region and background is changed to 0.
    Args:
        path: the path of the mask image
              type: String
        hash: the hash of mask
              type: String

    Return:
        The new grayscale mask image that only contains cilia
    '''
    mask = imread(path+hash+".png")
    newMask = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(mask[i][j]==2):
                newMask[i][j]=int(1)

    return newMask
