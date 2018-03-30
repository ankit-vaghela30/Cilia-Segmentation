import hastings.io_support as io
import hastings.optical_flow as of
import numpy as np

video_path = '../train/img_data/22e2bda8d3051a3ec1e2a6ee8834301b8587092b9e4a43afccdb4b6bf84ff727'
video = io.load_video(video_path)

def test_optical_flow():

    mask = of.optical_flow(video)
    assert np.shape(mask) == (256, 256)
