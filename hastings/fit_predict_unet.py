import numpy as np
from preprocessing import augment_data
from unet_model import unet
import argparse

#Importing keras function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K


#getting command line argument
parser = argparse.ArgumentParser(description='paths')
parser.add_argument('--mode',type=str,help='fit or predict?')
parser.add_argument('--image_path',type=str,help='path of image npy array')
parser.add_argument('--mask_path',type=str,help='path of mask npy array')
parser.add_argument('--model_path',type=str,help='path of saved model')
parser.add_argument('--save_path',type=str,help='path to save model or prediction')
args = parser.parse_args()

if(args.mode=="fit"):
    images,mask = augment_data(args.image_path,args.mask_path,args.mode)
if(args.mode=="predict"):
    images = augment_data(args.image_path,args.mask_path,args.mode)

def fit(training_images,mask_images,save_path):
    model = unet()
    #Fitting and saving model
    model.fit(training_images, mask_images, batch_size=4, epochs=100, verbose=1, shuffle=True)
    model.save(save_path+"/"+"model.h5")

def predict(testing_images,model_path,save_result_path):
    #loading model and predicting mask
    model=unet()
    model.load_weights(model_path)

    prediction = model.predict(testing_images, batch_size=4,verbose=1)
    np.save(save_result_path+'/prediction.npy', prediction)

if(args.mode=="fit"):
    fit(images,mask,args.save_path)
if(args.mode=="predict"):
    predict(images,args.model_path,args.save_path)
