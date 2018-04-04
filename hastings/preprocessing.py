import cv2

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


def augment_data(image_path,mask_path):
    '''
    loads images and mask as a numpy array, reshapes images and mask to 4 channels
    and normalzes image data
    
    Args:
        image_path:the path of the npy array containing images
                   type: String
        mask_path:      the path to npy array containing masks
                   type: String
    Return:
        numpy array of augmented mask and image
    '''
    image_array = np.load(image_path)
    mask_array = np.load(mask_path)
    
    images = image_array[...,np.newaxis]
    masks = mask_array[...,np.newaxis]

    image_data_array = images.astype('float32')
    mask_images = masks.astype('float32')
    data_images = image_data_array - np.mean(image_data_array)
    data_images = data_images / np.std(image_data_array)

    return data_images, mask_images
