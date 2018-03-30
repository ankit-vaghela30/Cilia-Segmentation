import cv2
import os
import numpy as np

def load_img(img_parent_path,hash):
    '''
        Load one image
        '''
    img = cv2.imread(os.path.join(img_parent_path,hash+'.png'),0)
    return img

def load_all_video(hash_file_path,video_parent_path):
    '''
        Load all video from provided txt file that stores all the hashes
        
        Args:
            hash_file_path: path of the txt file that stores all the hashes
                            type: STRING
                            
            video_parent_path: parent path of the video
                               type: STRING
                               
        Returns:
            dict: A dictionery of hash and the video
                  type: dict, content: { hash1:video1, hash2:video2, ... }
        '''
    hash_file = open(hash_file_path, "r").read()
    hashes = hash_file.split('\n')[0:-1]

    dict = {}
    for hash in hashes:
        v = load_video(os.path.join(video_parent_path,hash))
        dict[hash] = v
    return dict

def load_video(folder):
    '''
        Load the video from provided folder name
        
        Arg:
            folder: folder name of the video is stored
                    type: STRING
            
        Return:
            video: the loaded video
                   type ndarray, shape: (dim1,dim2)
        '''
    video = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            video.append(img)
    return video

def mkdir(folder_name):
    '''
        Create the folder to store the output pictures
        
        Args:
            folder_name: name of the folder to create
                         type: STRING
        '''
    os.makedirs(folder_name+'/')

def save_img(folder_name,hash,img):
    '''
        Save one image with the given hash name to the given folder
        
        Args:
            folder_name: name of the folder for storing pictures
                         type: STRING
        
            hash: hash name for each picture
                  type: STRING
        
            img: the picture to store
                 type: ndarray shape:(dim1,dim2)
        '''
    cv2.imwrite(folder_name+'/'+hash+'.png',img)

def play(name,image):
    '''
        Play the video
        
        Arg:
            image: each frame of the video
                   type: ndarray, shape: (dim1,dim2)
        '''
    cv2.imshow(name,image)
    cv2.waitKey(10)

def show(name,image):
    '''
        Show the image
        
        Arg:
            image: each frame of the video
                   type: ndarray, shape: (dim1,dim2)
        '''
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour2mask(contour):
    '''
        Change the output contour image into the mask we need
        
        Arg:
            contour: the original contour image
                     type: ndarray, shape: (dim1,dim2)
        
        Return:
            contour: the transformed mask with 2 standing for pixels of cilia
                     type: ndarray, shape: (dim1,dim2)
        '''
    contour[contour == 255] = 2
    contour[contour == 0] = 255
    return contour

def overlap(mask,of_mask):
    '''
        Find the overlapping between ground truth and mask from optical flow
        
        Args:
            mask: ground truth mask
                  type: ndarray, shape: (dim1,dim2)
                  
            of_mask: mask from optical flow
                     type: ndarray, shape: (dim1,dim2)
                     
        Return:
            mask: mask with 4 labels
                  type: ndarray, shape: (dim1,dim2)
        '''
    overlap_m = (mask == of_mask)
    mask[overlap_m] = 3
    return mask




