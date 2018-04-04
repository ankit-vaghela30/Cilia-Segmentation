import cv2
import os
import numpy as np

def load_img(img_parent_path,hash):
    '''
        Load mask image
        
        Args:
            img_parent_path: parent path of the image
                             type: STRING
        
            hash: hash name of the mask image
                  type: STRING
        
        Return:
            img: the mask image
                 type: ndarray, shape: (dim1,dim2)
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


def modifyMask(path,mask_hash):
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
    newMask = load_img(path,mask_hash)
    newMask[newMask==1] = 0
    newMask[newMask==2] = 1
    
    return newMask

def make_data(data_path,hashword,numImages):
    '''
    Reads image from a folder and returns a list of a specific number of images
    which is specified by numImages
    Args:
        path:      the path of the folder consisting of image
                   type: String
        hash:      the folder name
                   type: String
        numImages: the number of images to be read from folder
                   type: int
    Return:
        list of read images from the folder
    '''
    images = sorted(os.listdir(data_path+"/"+hashword))
    images = images[0:numImages]
    total = len(images)
    imgs = []
    imgs_mask = []

    for image_name in images:
        img = load_img(data_path+"/"+hashword,image_name.split(".")[0])
        img_msk = modifyMask(data_path+"/masks/",hashword)
        img_msk = np.array([img_msk])
        img = np.array([img])
        imgs.append(img[0])
        imgs_mask.append(img_msk[0])
    
    return imgs,imgs_mask