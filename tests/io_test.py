import hastings.io_support as io
import numpy as np

hash = '0bd52e3901cbbf4e57bd8bde4e0f7d5ba7bfc2b3b1dc4ce0dbf437d2ad374c5a'
full_path = '../test/img_data/'

def test_load_all_video():
    
    all_video = io.load_all_video('../test.txt',full_path)
    
    assert len(all_video) == 114
    assert np.shape(all_video[hash]) == (100, 480, 640)

def test_load_one_video():
    
    video = io.load_video(full_path+hash)
    
    assert np.shape(video) == (100, 480, 640)


