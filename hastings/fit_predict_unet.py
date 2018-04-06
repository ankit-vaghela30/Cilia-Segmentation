import numpy as np
from preprocessing import augment_data, convert_images_gray2rgb
from unet_model import unet
import tiramisu_model as tiramisu
import argparse

#Importing keras function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

def fit(training_images, mask_images, save_path, model):
    '''
        This function trains the UNET model and saves it.

        Args:
            training_images: prepared training images
                             type: np.array

            mask_images: prepared mask images
                         type: np.array

            save_path: path where the model will be saved
                       type: STRING

            model: tiramisu or unet
                   type: STRING

    '''

    if model is 'tiramisu':
        model= tiramisu.create_tiramisu(3, nb_layers_per_block=[4,5,7,10,12,15], p=0.2, wd=1e-4)
        training_images= convert_images_gray2rgb(training_images)
        mask_images= convert_images_gray2rgb(mask_images)
    else:
        model = unet()
        
    model.fit(training_images, mask_images, batch_size=4, epochs=100, verbose=1, shuffle=True)
    model.save(save_path+"/"+"model.h5")

def predict(testing_images, model_path, save_result_path, model):
    '''
        This function predicts the test images and saves the predictions

        Args:
            testing_images: prepared testing images
                            type: np.array

            model_path: path to the model
                        type: STRING

            save_result_path: path where the predictions will be saved
                              type: STRING

            model: tiramisu or unet
                   type: STRING

    '''

    if model is 'tiramisu':
        model= tiramisu.create_tiramisu(3, nb_layers_per_block=[4,5,7,10,12,15], p=0.2, wd=1e-4)
        testing_images= convert_images_gray2rgb(training_images)
    else:
        model=unet()

    model.load_weights(model_path)

    prediction = model.predict(testing_images, batch_size=4,verbose=1)
    np.save(save_result_path+'/prediction.npy', prediction)

def call_dl_model(args):
    '''
        This function is used to prepare data and train or predict depending on the argument

        Args:
            args: passed from user's arguments
    '''

    if(args.mode=="fit"):
        images,mask = augment_data(args.image_path, args.mask_path, args.mode)
        fit(images, mask, args.save_path, args.model)

    if(args.mode=="predict"):
        images = augment_data(args.image_path,args.mask_path,args.mode)
        predict(images,args.model_path,args.save_path, args.model)
