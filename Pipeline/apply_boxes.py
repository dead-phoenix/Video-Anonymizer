import cv2

class Masker() :

    def mask_video(self, path_in, scene_boxes, scene_frames) :
        cap = cv2.VideoCapture(path_in)

        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        path_out = path_in[:-4]+'_masked.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

        ret = True
        while ret :
            frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, image = cap.read()
            boxes = get_boxes(frame_count, scene_boxes, scene_frames)
            for box in boxes :
                cv2.rectangle(image, box[:2], box[2:], color=(255,0,0), thickness=-1)
            out_video.write(image)
        
        return(path_out)

def get_scene_index(frame_count, scene_frames) :
    for i in range(len(scene_frames)-1) :
        if scene_frames[i]<=frame_count<scene_frames[i+1] :
            return(i)
    return(len(scene_frames)-1)

def get_boxes(frame_count, scene_boxes, scene_frames) :
    # print(scene_boxes)
    idx = get_scene_index(frame_count, scene_frames)
    return(scene_boxes[idx])