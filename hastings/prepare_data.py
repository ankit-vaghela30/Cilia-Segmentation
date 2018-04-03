import numpy as np
import argparse
from io_support import make_data, modifyMask, load_img
#getting command line argument
parser = argparse.ArgumentParser(description='paths')
parser.add_argument('--mode',type=str,help='train or test?')
parser.add_argument('--file_path',type=str,help='path of train file consisting hash')
parser.add_argument('--data_path',type=str,help='path of folder consisting of video folder')
parser.add_argument('--num_images',type=int,help='number of images to be read from folder')
parser.add_argument('--save_path',type=str,help='path to save result')
args = parser.parse_args()

data_file= args.file_path
data_path = args.data_path
numImages= args.num_images

def prepare_data(data_file,data_path,numImages):
    file = open(data_file)
    data_image=[]
    data_mask=[]
    i=0
    for hashword in file:
        i=i+1
        image, mask = make_data(data_path,hashword.split("\n")[0],16)
        data_image.extend(image)
        data_mask.extend(mask)
        print("--- Number of folders read: ",str(i))
    data_image=np.array(data_image)
    
    data_mask = np.array(data_mask)
    return data_image, data_mask

if(args.mode=="train"):
    train_data_image,train_data_mask = prepare_data(data_file,data_path,numImages)
    np.save(args.save_path+'train_data_mask.npy',train_data_mask)
    np.save(args.save_path+'train_data_image.npy',train_data_image)
    print('Files saved as train_data_image.npy and train_data_mask.npy')

elif(args.mode=="test"):
    test_data_image,test_data_mask = prepare_data(data_file,data_path,numImages)
    np.save(args.save_path+ 'test_data_mask.npy', test_data_mask)
    print('Files saved as test_data_image.npy and test_data_mask.npy')

else:
    print("Please Choose correct mode")
