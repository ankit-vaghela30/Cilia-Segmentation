import cv2
import numpy as np
import hastings.preprocessing as pr
import hastings.io_support as io

def optical_flow(video, blur=5, threshold=110):
    '''
        This is for computing the optical flow of a given video.
        And by thresholding the images, only areas with high motion speed will be kept.
        This part is modified from this tutorial: https://docs.opencv.org/3.3.1/db/d7f/tutorial_js_lucas_kanade.html
        
        Args:
            video: the input video
                   type: ndarray, shape: (frames,dim1,dim2)
        
            blur: size of median filter
                  type: INT
        
            threshold: the threshold
                       type: INT
        
        Return:
            final_contour: the contour image
                           type: ndarray, shape: (dim1, dim2)
        '''
    frame_inital = video[0]
    h = np.zeros_like(frame_inital)
    h = np.expand_dims(h, axis=2)
    tmp_f = frame_inital
    r = pr.gray2rgb(tmp_f)
    hsv = pr.rgb2hsv(r)
    hsv[...,1] = 255

    contour_imgs = 0
    for frame in video[1:]:
        io.play('frame1',frame)
        prev = frame_inital
        next = frame
        flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        bgr_mean = cv2.medianBlur(bgr,5)
        gray = cv2.cvtColor(bgr_mean,cv2.COLOR_BGR2GRAY)

        gray_img_thr = pr.threshold(gray,threshold)
        io.play('frame2',gray_img_thr)

        contour_imgs += np.asarray(gray_img_thr)
        frame_inital = next
    
    _,final_contour = cv2.threshold(contour_imgs,225,255,cv2.THRESH_BINARY)
    mask = io.contour2mask(final_contour)
    
    return final_contour

def of2prediction(output_folder,hash_path,data_folder,threshold_value=80):
    '''
        This is for doing cilia predictions with optical flow only.
        
        Args:
            output_folder: folder path where output will be stored in
                           type: STRING
        
            hash_path: path for hash file
                       type: STRING
        
            data_folder: path for folder where video data are stored in
                         type: STRING
        
            threshold_value: threshold value to be passed into optical flow
                             type: INT
        '''
    
    io.mkdir(output_folder)
    all_videos_dict = io.load_all_video(hash_path, data_folder)
    
    for hash in all_videos_dict:
        video = all_videos_dict[hash]
        mask = optical_flow(video,threshold=threshold_value)
        
        io.save_img(output_folder,hash,mask)
