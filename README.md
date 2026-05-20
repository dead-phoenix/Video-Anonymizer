# Volta-Medical


Drive link for access to v1  of annonymiser app: https://drive.google.com/drive/folders/1udYUDdQsFKpmF-1TM4d_tULyRFhJaG1W?usp=sharing


PyIntaller command to generate the exe file (on Windows)

python -m PyInstaller anonymisation_antoine.py --onefile --hidden-import ultralytics --collect-data ultralytics --collect-data en_core_web_sm --collect-data ru_core_news_sm --collect-data fr_core_news_sm --collect-data es_core_news_sm --collect-data pytesseract --name anonymizer-dev-v2 --add-data="C:\Program Files\Tesseract-OCR;Tesseract-OCR"



Required libraries :

WIP
