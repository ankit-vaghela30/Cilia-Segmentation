import argparse
from hasting.fit_predict_unet import call_dl_model
from hasting.optical_flow import of2prediction
from hasting.prepare_data import call_prepare
import io_support, optical_flow, preprocessing, unet_model

def main():
    #getting command line argument
    parser = argparse.ArgumentParser(description='paths')
    parser.add_argument('--model',type=str,help='unet, optical_flow or tiramisu')
    parser.add_argument('--mode',type=str,help='fit or predict?')
    parser.add_argument('--operation',type=str,help='Prepare data or use models')
    parser.add_argument('--image_path',type=str,help='path of image npy array')
    parser.add_argument('--mask_path',type=str,help='path of mask npy array')
    parser.add_argument('--model_path',type=str,help='path of saved model')
    parser.add_argument('--save_path',type=str,help='path to save model or prediction
    parser.add_argument('--file_path',type=str,help='path of train file consisting hash')
    parser.add_argument('--data_path',type=str,help='path of folder consisting of video folder')
    parser.add_argument('--num_images',type=int,help='number of images to be read from folder')

    args = parser.parse_args()

    if args.operation is 'preprocess':
        print('Preprocessing and saving the data')
        call_prepare(args)

    if args.model is 'unet' or args.model is 'tiramisu':
        print('Deep learning model is called ', args.model)
        call_dl_model(args)

    if args.model is 'optical_flow':
        print('optical_flow model is called')
        of2prediction(args.save_path, args.file_path, args.data_path)
