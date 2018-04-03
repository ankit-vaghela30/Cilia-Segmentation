import hastings.io_support as io
import numpy as np

hash = '22e2bda8d3051a3ec1e2a6ee8834301b8587092b9e4a43afccdb4b6bf84ff727'
full_path = '../train/img_data/'
mask = '../masks/'

m_img = np.asarray([[0,0,0],[1,0,2],[0,1,2]])
of_contour = np.asarray([[0,0,0],[0,255,255],[0,0,255]])
of_img = np.asarray([[255,255,255],[255,2,2],[255,255,2]])

def test_load_all_video():
    
    all_video = io.load_all_video('../train.txt',full_path)
    assert len(all_video) == 211
    assert np.shape(all_video[hash]) == (100, 256, 256)

def test_load_one_video():
    
    video = io.load_video(full_path+hash)
    assert np.shape(video) == (100, 256, 256)

def test_load_img():

    mask = io.load_img(mask,hash)
    assert np.shape(mask) == (256, 256)

def test_contour2mask():

    mask = io.contour2mask(of_contour)
    assert mask == of_img

def test_overlap():

    mask = io.overlap(m_img,of_img)
    assert np.shape(mask) == (3, 3)

