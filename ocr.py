import pytesseract
import cv2

from PIL import Image

def do_ocr(image_path, language):
    # image = cv2.imread(image_path)
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=language)
    # Truncate OCR text if it's too long
#     max_ocrtext_length = 255 
#     if len(text) > max_ocrtext_length:
#          text = text[:max_ocrtext_length]
    return text


# text = do_ocr(r'C:\Article-Detection-App\ARTICLE_IMAGES\articlecrops\predict6259\crops\Article\HT_Del_091224_10_10\predict\crops\story\abc.jpg', 'eng')

# print(text)

def image_ocr(image, language):
   
    text = pytesseract.image_to_string(image, lang=language)
    return text