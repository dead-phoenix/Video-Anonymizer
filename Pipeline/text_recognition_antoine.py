import spacy
from langdetect import detect
import pytesseract
import re
import cv2
import time

import en_core_web_sm
import fr_core_news_sm
import ru_core_news_sm
import es_core_news_sm

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class TextFiltering:
    def __init__(self, models_dir):
        # self.nlp_en = spacy.load("en_core_web_sm")
        # self.nlp_fr = spacy.load("fr_core_news_sm")
        # self.nlp_ru = spacy.load("ru_core_news_sm")
        # self.nlp_es = spacy.load("es_core_news_sm")
        self.nlp_en = en_core_web_sm.load()
        self.nlp_fr = fr_core_news_sm.load()
        self.nlp_ru = ru_core_news_sm.load()
        self.nlp_es = es_core_news_sm.load()

    def detect_language_and_extract_names(self, text):
        text = re.sub(r'[^\w\s.,;\'"\-\—]', '', text) 
        try:
            language = detect(text)

            nlp = self.nlp_en
            if language in ("fr", "ru", "es"):
                nlp = {"fr": self.nlp_fr, "ru": self.nlp_ru, "es": self.nlp_es}[language]

            doc = nlp(text)
            potential_names = [token.text for token in doc if token.is_title]

            person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            all_names = set(potential_names) | set(person_names)
            return list(all_names)
        except Exception as e:
            print(f"Error in processing text: {text}")
            print(f"Error details: {str(e)}")
            return []

    def process_boxes(self, image, boxes):
        results = []
        results_with_names = []
        for (startX, startY, endX, endY) in boxes:
            # Assuming boxes is not None, add additional checks if needed
            startX = int(startX)
            startY = int(startY)
            endX = int(endX)
            endY = int(endY)

            # Extract the region of interest
            r = image[startY:endY, startX:endX]
            # print(startX, endX, startY, endY)

            configuration = ("-l eng --oem 1 --psm 8")
            text = pytesseract.image_to_string(r, config=configuration)

            # Skip processing if text contains only numeric values
            if not text.strip().isdigit():
                person_names = self.detect_language_and_extract_names(text)
                if person_names:
                    #person_names_str = ', '.join(person_names)
                    #results.append(((startX, startY, endX, endY), person_names_str))
                    results.append((startX, startY, endX, endY))
                    results_with_names.append((startX, startY, endX, endY, person_names))

        return results, results_with_names
    
    def process_scene(self, path, boxes_scene, scene_frames) :
        cap = cv2.VideoCapture(path)

        dict_boxes_filtered = {}
        dict_boxes_names = {}

        for k, (boxes, frame_num) in enumerate(zip(boxes_scene.values(), scene_frames)) :
            cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_num)
            _, image = cap.read()
            boxes_filtered, boxes_with_names = self.process_boxes(image, boxes)
            dict_boxes_filtered[k] = boxes_filtered
            dict_boxes_names[k] = boxes_with_names
        
        return(dict_boxes_filtered, dict_boxes_names)
