from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop

import cv2

class Splitter() :

    def split_video(self, input_path, main_dir) :

        file_name = input_path.replace('\\', '/').rsplit("/", 1)[-1].rsplit(".", 1)[0]

        video = VideoFileClip(input_path)
        width, height = video.size

        split_width = width // 2
        split_height = height // 2

        splits = [
            (0, 0, split_width, split_height),
            (split_width, 0, width, split_height),
            (0, split_height, split_width, height),
            (split_width, split_height, width, height)
        ]

        locs = ['ul', 'ur', 'bl', 'br']
        subvid_names = {}

        for loc, (x1, y1, x2, y2) in zip(locs, splits):
            subclip = video.subclip(0, video.duration)
            subclip = subclip.fx(crop, x1=x1, y1=y1, x2=x2, y2=y2)

            subclip.write_videofile(main_dir+f'/temp_anon/{file_name}_{loc}.mp4', codec='libx264', audio_codec='aac', logger=None)

            subvid_names[loc] = (main_dir+f'/temp_anon/{file_name}_{loc}.mp4')

        video.close()

        return(subvid_names)
    
    def split_with_cv(self, input_path, main_dir) :

        file_name = input_path.replace('\\', '/').rsplit("/", 1)[-1].rsplit(".", 1)[0]

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out_ul = cv2.VideoWriter(main_dir+f'/temp_anon/{file_name}_ul.mp4', fourcc, fps, (width//2, height//2))
        out_ur = cv2.VideoWriter(main_dir+f'/temp_anon/{file_name}_ur.mp4', fourcc, fps, (width//2, height//2))
        out_bl = cv2.VideoWriter(main_dir+f'/temp_anon/{file_name}_bl.mp4', fourcc, fps, (width//2, height//2))
        out_br = cv2.VideoWriter(main_dir+f'/temp_anon/{file_name}_br.mp4', fourcc, fps, (width//2, height//2))

        ret = True
        while ret :
            ret, image = cap.read()
            if ret :
                out_ul.write(image[:height//2, :width//2])
                out_ur.write(image[height//2:, :width//2])
                out_bl.write(image[:height//2, width//2:])
                out_br.write(image[height//2:, width//2:])
        
        cap.release()
        out_ul.release()
        out_ur.release()
        out_bl.release()
        out_br.release()

        locs = ['ul', 'ur', 'bl', 'br']
        subvid_names = {}
        for loc in locs :
            subvid_names[loc] = (main_dir+f'/temp_anon/{file_name}_{loc}.mp4')

        return(subvid_names)