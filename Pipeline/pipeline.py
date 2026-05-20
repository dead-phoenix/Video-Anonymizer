from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop
from moviepy.editor import clips_array

import time
t_init_total = time.time()

import os
import glob

from pathlib import Path 

import cv2
import numpy as np
import openvino as ov
from PIL import Image

from scenedetect import detect, ContentDetector

from argparse import ArgumentParser 

ap = ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, default='output.mp4',
	help="path to output video")
args = vars(ap.parse_args())

MAIN_DIR = os.getcwd()
print('Working in '+MAIN_DIR)

if not os.path.exists(MAIN_DIR+'/temp_anon') :
    os.mkdir(MAIN_DIR+'/temp_anon')
else :
    files = glob.glob(MAIN_DIR+'/temp_anon/*')
    for file in files :
        os.remove(file)

###################################
# Function to split in subscreens using moviepy
###################################

def split_video(input_path):

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

    idxs = ['11', '12', '21', '22']

    for idx, (x1, y1, x2, y2) in zip(idxs, splits):
        subclip = video.subclip(0, video.duration)
        subclip = subclip.fx(crop, x1=x1, y1=y1, x2=x2, y2=y2)

        subclip.write_videofile(MAIN_DIR+f'/temp_anon/cropped_{idx}.mp4', codec='libx264', audio_codec='aac')

    video.close()

###################################
# OpenVINO setup
###################################

core = ov.Core()

model_dir = Path("/home/volta/scene_detection/model")
precision = "FP16"
detection_model = "horizontal-text-detection-0001"
recognition_model = "text-recognition-resnet-fc"

model_dir.mkdir(exist_ok=True)

detection_model_path = (model_dir / "intel/horizontal-text-detection-0001" / precision / detection_model).with_suffix(".xml")
recognition_model_path = (model_dir / "public/text-recognition-resnet-fc" / precision / recognition_model).with_suffix(".xml")

detection_model = core.read_model(
    model=detection_model_path, weights=detection_model_path.with_suffix(".bin")
)
detection_compiled_model = core.compile_model(model=detection_model, device_name='CPU')

detection_input_layer = detection_compiled_model.input(0)

recognition_model = core.read_model(
    model=recognition_model_path, weights=recognition_model_path.with_suffix(".bin")
) 

recognition_compiled_model = core.compile_model(model=recognition_model, device_name='CPU')

recognition_output_layer = recognition_compiled_model.output(0)
recognition_input_layer = recognition_compiled_model.input(0)


###################################
# Functions for text detection and recognition
###################################

def multiply_by_ratio(ratio_x, ratio_y, box):
    return [
        max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
        for idx, shape in enumerate(box[:-1])
    ]


def run_preprocesing_on_crop(crop, net_shape):
    temp_img = cv2.resize(crop, net_shape)
    temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
    return temp_img


def do_detection(image, W, H) :
    resized_image = cv2.resize(image, (W, H))
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    output_key = detection_compiled_model.output("boxes")
    boxes = detection_compiled_model([input_image])[output_key]
    boxes = boxes[~np.all(boxes == 0, axis=1)]

    return(resized_image, boxes)


def do_recognition(image, resized, boxes, letters, W, H) :
    (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    annotations = list()
    cropped_images = list()

    for i, crop in enumerate(boxes):
        (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
        image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))

        result = recognition_compiled_model([image_crop])[recognition_output_layer]

        recognition_results_test = np.squeeze(result)

        annotation = list()
        for letter in recognition_results_test:
            parsed_letter = letters[letter.argmax()]

            if parsed_letter == letters[0]:
                break
            annotation.append(parsed_letter)
        annotations.append("".join(annotation))
        cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
        cropped_images.append(cropped_image)

    boxes_with_annotations = list(zip(boxes, annotations))

    return(boxes_with_annotations)


def filter_boxes(boxes, img, resized_img, thresh) :
    (real_y, real_x), (resized_y, resized_x) = img.shape[:2], resized_img.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    res = []
    for box in boxes :
        if box[-1]>thresh :
            res.append(list(map(int, multiply_by_ratio(ratio_x, ratio_y, box))))
    
    return(res)


def generate_boxes(video, begin_frames, Hd, Wd, Hr, Wr, letters, thresh=0.3) :
    boxes_dict = {}
    for k, frame in enumerate(begin_frames) :
        # print(frame)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = video.read()
        resized, boxes = do_detection(img, Hd, Wd)
        # boxes_with_annotations = do_recognition(img, resized, boxes, letters, Wr, Hr)
        boxes_dict[str(k)] = filter_boxes(boxes, img, resized, thresh)
    return(boxes_dict)

###################################
# Functions for scene detection
###################################

def find_index(i, limits) :
    for k, (mini, maxi) in enumerate(limits) :
        if mini<=i<maxi :
            return(k)


def detect_scenes(file) :
    scene_list = detect(file, ContentDetector(), start_in_scene=True)
    edges_list = [[scene[0].get_frames(), scene[1].get_frames()] for scene in scene_list]
    begin_frames = [scene[0].get_frames() for scene in scene_list]
    return(edges_list, begin_frames)

###################################
# Splitting in 4 subscreens
###################################

input_path = args['input']

ti = time.time()
subscreens = split_video(input_path)
print(f"Elapsed time for split : {time.time()-ti:.2f}s")

###################################
# Apply text detection on the subscreens
###################################

ti = time.time()
cropped_files = [MAIN_DIR+'/temp_anon/cropped_11.mp4',
                 MAIN_DIR+'/temp_anon/cropped_12.mp4',
                 MAIN_DIR+'/temp_anon/cropped_21.mp4',
                 MAIN_DIR+'/temp_anon/cropped_22.mp4']

for cropped_file in cropped_files :

    print(f"Treating file "+cropped_file.rsplit('/', 1)[-1])

    # OpenCV reading input video and creating output video object

    vid = cv2.VideoCapture(cropped_file)
    fps = vid.get(cv2.CAP_PROP_FPS)
    nbr_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(f"Video has {nbr_frame} images at {fps} FPS, format {height} x {width}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(cropped_file[:-6]+'boxes_'+cropped_file[-6:], fourcc, fps, (width, height))

    # Scene detection

    edges_list, begin_frames = detect_scenes(cropped_file)

    # Text detection and recognition


    N, C, Hd, Wd = detection_input_layer.shape
    _, _, Hr, Wr, = recognition_input_layer.shape
    letters = "~0123456789abcdefghijklmnopqrstuvwxyz"

    boxes_dict = generate_boxes(vid, begin_frames, Hd, Wd, Hr, Wr, letters)

    # Writing of the output video

    frame_number = -1

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while(frame_number<nbr_frame):
        frame_number += 1
        ret, frame = vid.read()
        if ret :
            idx = find_index(frame_number, edges_list)
            for box in boxes_dict[str(idx)] :
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=-1)
            out_video.write(frame)

    vid.release()
    out_video.release()

    os.remove(cropped_file)

    print(f"{frame_number}/{nbr_frame} for {cropped_file.rsplit('/', 1)[-1]}")

print(f"Elapsed time for boxes : {time.time()-ti:.2f}s")

###################################
# Concatenate all the boxed files
###################################

ti = time.time()
boxed_files = [MAIN_DIR+'/temp_anon/cropped_boxes_11.mp4',
               MAIN_DIR+'/temp_anon/cropped_boxes_12.mp4',
               MAIN_DIR+'/temp_anon/cropped_boxes_21.mp4',
               MAIN_DIR+'/temp_anon/cropped_boxes_22.mp4']
print(boxed_files)
clips = np.array(list(map(VideoFileClip, boxed_files))).reshape(2, 2)
final = clips_array(clips)

final.write_videofile(MAIN_DIR+'/'+args['output'], codec='libx264', audio_codec='aac')

for boxed_file in boxed_files :
    os.remove(boxed_file)
os.rmdir(MAIN_DIR+'/temp_anon')

print(f"Elapsed time for concatenate : {time.time()-ti:.2f}s")

print(f"Total time : {time.time()-t_init_total:.2f}s")