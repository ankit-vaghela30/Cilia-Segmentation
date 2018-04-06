import hastings.io_support as io
import hastings.preprocessing as pre
import numpy as np


video_path = '../test/img_data/0bd52e3901cbbf4e57bd8bde4e0f7d5ba7bfc2b3b1dc4ce0dbf437d2ad374c5a'
npy_image_path = "../test/npy_data/test_data.npy"
#>>>>>>> master
video = io.load_video(video_path)
frame = video[0]

def test_threshold():

    t_video = pre.threshold(frame,100)
    assert np.shape(t_video) == (256, 256)

def test_blur():

    b_video = pre.blur(frame,5)
    assert np.shape(b_video) == (256, 256)

def test_convertion():

    rgb = pre.gray2rgb(frame)
    assert np.shape(rgb) == (256, 256, 3)

    gray = pre.rgb2gray(rgb)
    assert np.shape(gray) == (256, 256)

    hsv = pre.rgb2hsv(rgb)
    assert np.shape(hsv) == (256, 256, 3)

    h2rgb = pre.hsv2rgb(hsv)
    assert np.shape(h2rgb) == (256, 256, 3)

def test_augment_data(npy_image_path,None,"predict"):
    images = pre.augment_data(npy_image_path,None,"predict")
    assert np.shape(images) == (1824,256,256,1)

def test_convert_images_gray2rgb(npy_image_path):
    images = np.load(npy_image_path)
    rgb_images = pre.convert_images_gray2rgb(images)
    assert np.shape(rgb_images) == (images.shape[0], 256, 256, 3)
