import numpy as np
import argparse
from io_support import make_data, modifyMask, load_img

def prepare_data(data_file,data_path,numImages,mode):
    '''
        This function will creaet npy array for given images

        args:
            data_file: path of train file consisting hash
                       type: STRING

            data_path: path of folder consisting of video folder
                       type: STRING

            numImages: number of images to be read from folder
                       type: INT

            mode: test or train
                  type:

        return:
                depending on mode, returns a numpy array of all images or images and masks
    '''

    file = open(data_file)
    data_image=[]
    data_mask=[]
    i=0
    for hashword in file:
        i=i+1
        if (mode=="train"):
            image, mask = make_data(data_path,hashword.split("\n")[0],numImages,mode)
            data_image.extend(image)
            data_mask.extend(mask)
        if(mode=="test"):
            image = make_data(data_path,hashword.split("\n")[0],numImages,mode)
            data_image.extend(image)
        print("--- Number of folders read: ",str(i))
    data_image=np.array(data_image)

    if (mode=="train"):
        data_mask = np.array(data_mask)
        return data_image, data_mask
    else:
        return data_image

def call_prepare(args):
    '''
        This function is used to create appropriate data

        Args:
            args: passed from user's arguments

    '''
    data_file= args.file_path
    data_path = args.data_path
    numImages= args.num_images

    if(args.mode=="train"):
        train_data_image,train_data_mask = prepare_data(data_file,data_path,numImages,args.mode)
        np.save(args.save_path+'train_data_mask.npy',train_data_mask)
        np.save(args.save_path+'train_data_image.npy',train_data_image)
        print('Files saved as train_data_image.npy and train_data_mask.npy')

    elif(args.mode=="test"):
        test_data_image = prepare_data(data_file,data_path,numImages,args.mode)
        np.save(args.save_path+ 'test_data.npy', test_data_image)
        print('Files saved as test_data_image.npy')

    else:
        print("Please Choose correct mode")
