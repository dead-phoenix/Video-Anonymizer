from multiprocessing import freeze_support

import os
from shutil import rmtree
import time

from split_screen import Splitter
from scene_processing import SceneDetector
from text_detector import EastDetector, detect_text_on_video, YoloDetector
from apply_boxes import Masker
from merge_screen import Merger

from gooey import Gooey, GooeyParser

freeze_support()

@Gooey(program_name="Anonymisation")
def get_args():

    parser = GooeyParser(description="Anonymiser des vidéos")

    parser.add_argument(
        "--input",
        metavar="Video d'entrée",
        help="",
        widget='FileChooser',
        type=str,
        required=True,
        default="",
    )

    parser.add_argument(
        "--output",
        metavar="Nom de la vidéo anonymisée",
        help="Enregistrée au même endroit que la vidéo traitée",
        type=str,
        default="output.mp4",
        required=True
    )

    parser.add_argument(
        "--do_east",
        metavar="Utiliser EAST",
        help="Inclure EAST dans les modèles à utiliser",
        widget='BlockCheckbox',
        action='store_true'
    )

    parser.add_argument(
        "--do_yolo",
        metavar="Utiliser YOLO",
        help="Inclure YOLO dans les modèles à utiliser",
        widget='BlockCheckbox',
        action='store_true'
    )

    parser.add_argument(
        "--thresh_east",
        metavar="Seuil pour le modèle EAST",
        help="Doit être entre 0 et 1 (défault : 0.3)",
        widget='DecimalField',
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--thresh_yolo",
        metavar="Seuil pour le modèle YOLO",
        help="Doit être entre 0 et 1 (défault : 0.2)",
        widget='DecimalField',
        type=float,
        default=0.2,
        required=False
    )

    return parser.parse_args()

thresh_scene = 10.0
thresh_recogn = 0.5
thresh_ner = 0.5

class Info() :
    def __init__(self, path) :
        self.input=path

def make_working_dir(path) :
    path_temp = path+'temp_anon' if  path.endswith('/') else path+'/temp_anon'
    if not os.path.exists(path_temp) :
        os.mkdir(path_temp)

def remove_working_dir(path) :
    rmtree(path)

def main() :
    tinit = time.time()

    freeze_support()

    args = get_args()

    input_file = args.input
    output_file = args.output if args.output.endswith('.mp4') else args.output+'.mp4'
    thresh_east = args.thresh_east
    thresh_yolo = args.thresh_yolo

    main_dir = input_file.replace('\\', '/').rsplit('/', 1)[0]          # main_dir is the directory of the input video

    make_working_dir(main_dir)                                          # create a temporary directory temp_ano in main_dir
    
    info = Info(input_file)

    # Splitting
    print('Splitting')
    ti = time.time()
    subscreens_path = Splitter().split_with_cv(info.input, main_dir)
    info.subscreens = subscreens_path
    print(f'Splitting done in {time.time()-ti:.3f}s')
    
    # Scene detection
    print('Scene detection')
    ti = time.time()
    frames_dict = {}
    for loc, subscreen in info.subscreens.items() :
        frames = SceneDetector().detect_scenes(subscreen, thresh_scene)
        frames_dict[loc] = frames
    info.scene_frames = frames_dict
    print(f'Scene detection done in {time.time()-ti:.3f}s')
    
    # Text detection
    print('Text detection')
    ti = time.time()
    boxes_dict = {}
    models = {}
    if args.do_east :
        east_model = EastDetector(conf=thresh_east, model_path=os.getcwd().replace('\\', '/')+'/models/frozen_east_text_detection.pb', width=640, height=640)
        models['EAST'] = east_model
    if args.do_yolo :
        yolo_model = YoloDetector(conf=thresh_yolo, model_path=os.getcwd().replace('\\', '/')+'/models/best-xl-e100.pt', width=640, height=640)
        models['YOLO'] = yolo_model
    if len(models)==0 :
        print('##########\nAttention, pas de modèle sélectionné\nLe programme va finir mais ne fera rien\n##########')
    for loc, subscreen in info.subscreens.items() :
        boxes = detect_text_on_video(path=subscreen, frames=info.scene_frames[loc], models=models)
        boxes_dict[loc] = boxes
    info.boxes = boxes_dict
    print(f'Text detection done in {time.time()-ti:.3f}s')

    # result.text_recogn()
    # result.do_ner()

    # Apply mask
    print('Masking')
    ti = time.time()
    masked_subscreens = {}
    for loc, subscreen in info.subscreens.items() :
        masked_subscreen = Masker().mask_video(subscreen, info.boxes[loc], info.scene_frames[loc])
        masked_subscreens[loc] = masked_subscreen
    info.masked_subscreens = masked_subscreens
    print(f'Masking in {time.time()-ti:.3f}s')
    
    # Merge masked subscreens
    print('Merging')
    ti = time.time()
    output_path = Merger().merge_with_cv(info.masked_subscreens, output_file, main_dir)
    info.output_path = output_path
    print(f'Merging done in {time.time()-ti:.3f}s')

    remove_working_dir(main_dir+'/temp_anon')

    print(f'Total execution time : {time.time()-tinit:.3f}s')

if __name__=="__main__" :
        freeze_support()
        main()