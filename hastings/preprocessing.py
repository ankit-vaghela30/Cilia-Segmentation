import cv2

def threshold(frame,t):
    '''
        Setting threhold to the video. Turn pixels below threshold black.
        
        Args:
            video: the input grayscale video
                   type: ndarray, shape: (dim1,dim2)
            t: the threshold
               type: INT
        
        Return:
            new_video: the transformed video
        '''
    _, new_frame = cv2.threshold(frame,t,255,cv2.THRESH_BINARY)
    return new_frame

def blur(video,b):
    '''
        Smoothing the video
        
        Args:
            video: the input video
                   type: ndarray, shape: (dim1,dim2)
            b: size of median filter
            
        Return:
            the transformed video
        '''
    return cv2.medianBlur(video,b)

def gray2rgb(video):
    '''
        Transform grayscale video to RGB video
        
        Args:
            video: the input grayscale video
                   type: ndarray, shape: (dim1,dim2)
                   
        Return:
            the transformed RGB video
        '''
    return cv2.cvtColor(video,cv2.COLOR_GRAY2RGB)

def rgb2gray(video):
    '''
        Transform RGB video to grayscale video
        
        Args:
            video: the input RGB video
                   type: ndarray, shape: (dim1,dim2)
                   
        Return:
            the transformed grayscale video
        '''
    return cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)

def rgb2hsv(video):
    '''
        Transform BGR video to HSV video
        
        Args:
            video: the input RGB video
                   type: ndarray, shape: (dim1,dim2)
        
        Return:
            the transformed HSV video
        '''
    return cv2.cvtColor(video,cv2.COLOR_BGR2HSV)

def hsv2rgb(video):
    '''
        Transform HSV video to RGB video
        
        Args:
            video: the input HSV video
                   type: ndarray, shape: (dim1,dim2)
        
        Return:
            the transformed RGB video
        '''
    return cv2.cvtColor(video,cv2.COLOR_HSV2BGR)











