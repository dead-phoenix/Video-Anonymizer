from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import clips_array

import cv2

import numpy as np

class Merger() :

    def merge_files(self, files, out, main_dir) :
        list_files = list(files.values())
        masked_clips = np.array(list(map(VideoFileClip, list_files))).reshape(2, 2)
        final = clips_array(masked_clips)
        final.write_videofile(main_dir+'/'+out, codec='libx264', audio_codec='aac', logger=None)

        return(main_dir+'/'+out)
    
    def merge_with_cv(self, files, out, main_dir) :
        cap_ul = cv2.VideoCapture(files['ul'])
        cap_ur = cv2.VideoCapture(files['ur'])
        cap_bl = cv2.VideoCapture(files['bl'])
        cap_br = cv2.VideoCapture(files['br'])

        hu = cap_ul.get(cv2.CAP_PROP_FRAME_HEIGHT)
        hb = cap_bl.get(cv2.CAP_PROP_FRAME_HEIGHT)
        wl = cap_ul.get(cv2.CAP_PROP_FRAME_WIDTH)
        wr = cap_ur.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap_br.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        cap_out = cv2.VideoWriter(main_dir+'/'+out, fourcc, fps, (int(wl+wr), int(hu+hb)))

        ret = True
        while ret :
            ret, image_ul = cap_ul.read()
            _, image_ur = cap_ur.read()
            _, image_bl = cap_bl.read()
            _, image_br = cap_br.read()
            if ret :
                image_out = np.vstack((np.hstack((image_ul,image_bl)), np.hstack((image_ur, image_br))))
                cap_out.write(image_out)
        
        cap_ul.release()
        cap_ur.release()
        cap_bl.release()
        cap_br.release()
        cap_out.release()

        return(main_dir+'/'+out)