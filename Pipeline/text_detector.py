import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from ultralytics import YOLO

import time

class EastDetector() :

    def __init__(self, model_path, conf=0.5, width=640, height=640) :
        self.model = model_path
        self.conf = conf
        self.w = width
        self.h = height
    
    def get_output_layers(self) :
        image = self.input              # changed cv2.imread(self.input) to self.input as self.input already numpy array of the image

        origH, origW = image.shape[:2]

        rW = origW / float(self.w)
        rH = origH / float(self.h)

        image = cv2.resize(image, (self.w, self.h))
        H, W = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)

        net = cv2.dnn.readNet(self.model)

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        
        net.setInput(blob)
        scores, geometry = net.forward(layerNames)   

        return(scores, geometry, rW, rH)
    

    def predictions(self, prob_score, geo) :
        numR, numC = prob_score.shape[2:4]
        boxes = []
        confidence_val = []

        for y in range(0, numR) :
            scoresData = prob_score[0, 0, y]
            x0 = geo[0, 0, y]
            x1 = geo[0, 1, y]
            x2 = geo[0, 2, y]
            x3 = geo[0, 3, y]
            anglesData = geo[0, 4, y]

            for i in range(0, numC) :
                if scoresData[i] < self.conf :
                    continue

                (offX, offY) = (i * 4.0, y * 4.0)

                angle = anglesData[i]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = x0[i] + x2[i]
                w = x1[i] + x3[i]   

                endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
                endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
                startX = int(endX - w)
                startY = int(endY - h)

                boxes.append((startX, startY, endX, endY))
                confidence_val.append(scoresData[i])

        return (boxes, confidence_val)
    
    
    def get_boxes(self, inp) :
        self.input = inp
        scores, geometry, rW, rH = self.get_output_layers()
        boxes, confidence_val = self.predictions(scores, geometry)

        boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
        res = []
        for (sx, sy, ex, ey) in boxes :
            res.append((int(max(sx*rW, 0)), int(max(sy*rH, 0)), int(max(ex*rW, 0)), int(max(ey*rH, 0))))

        return(res)

class YoloDetector() :
    
    def __init__(self, model_path, conf=0.25, width=640, height=640) :
        self.model = model_path
        self.conf = conf
        self.w = width
        self.h = height

    def get_boxes(self, inp) :
        self.input = inp
        yolom = YOLO(self.model)
        results = yolom(self.input, imgsz=[self.h, self.w], conf=self.conf, device='cpu', verbose=False)

        boxes = results[0].boxes.xyxy
        res = []
        for (sx, sy, ex, ey) in boxes :
            res.append((int(sx), int(sy), int(ex), int(ey)))

        return(res)


def detect_text_on_video(path, frames, models) :
    cap = cv2.VideoCapture(path)

    boxes_dict = {}

    for k, frame in enumerate(frames) :

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()

        boxes = []
        for name, model in models.items() :
            ti = time.time()
            boxes += model.get_boxes(img)
            print(name+f' done in {time.time()-ti:.2f}s')
        
        boxes_dict[k] = boxes
    
    return(boxes_dict)

