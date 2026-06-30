import os
import re

import requests
import shutil
import base64

import subprocess
import mysql.connector
# import nltk
from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
from transformers import pipeline, AutoTokenizer, CLIPProcessor, CLIPModel
from datetime import datetime
from keywords import extract_keywords
from ocr import do_ocr, image_ocr
from img_dim import get_image_dimensions as gid
#import scrape as scr
import title as tle 
# import envchange as envc
import time
import shutil
# import pdf_TO_image as pti
from textblob import TextBlob
import mbart as mbart
# from mr_sentiment import get_marathi_sentiment
from hocr import do_hocr
from story import extract_story_from_soup
from compare_conf import compare_conf
from image_crop import generate_cropped_image , yolo_to_pixel_coords
from mask_image import masked_image

from logos_match import match_logo

from grayscale import convert_to_grayscale

from advetisement_data_using_VLM import identify_unknown_brand_using_VLM
import json



# for calssifier
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class NewspaperCropClassifier:
    def __init__(self, model_path="newspaper_convnext2_best.pt"):
        self.class_names = {0: "Article", 1: "Classified", 2: "Display"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Architecture & Weights
        self.model = models.convnext_tiny()
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 3)
        
        # We use weights_only=True for safe loading, resolving standard PyTorch warnings
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        self.model.eval()
        self.model.to(self.device)

        # Standard Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """Takes a single image path and returns the predicted class and confidence."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                max_prob, predicted_idx = torch.max(probabilities, 0)

            confidence_pct = int(round(max_prob.item() * 100))
            predicted_class = self.class_names[predicted_idx.item()]

            return {"predicted_class": predicted_class, "confidence": confidence_pct}
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


from pathlib import Path

import logging
from logging.handlers import RotatingFileHandler

# 1. Get the absolute path to the folder containing THIS script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, "app.log")

# 1. Setup: This single block configures everything
logging.basicConfig(
    level=logging.WARNING, # Capture everything from DEBUG up to CRITICAL
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        RotatingFileHandler(log_path, maxBytes=10**6, backupCount=3), # 1MB limit, 3 backups
        logging.StreamHandler() # Also print to console for live debugging
    ]
)

logger = logging.getLogger(__name__) # Use module-level logger
logger.setLevel(logging.DEBUG)

import random

from tensorflow.python.keras.engine import data_adapter

#-----------------imports for xml file-----------------#
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

#-----------------imports for categorization------------#
from article_categorization import categorization

#-----------------import for email finder-----------------#
from email_finder import separate_email_from_text

#-----------------import for article dimension-----------------#
from image_dimension import dimension_acc_newspaper

#-------------------low resolution-----------------------------#
from low_res import create_low_res

# -------------------for text below image----------
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
import textwrap
from multiprocessing import Pool

# Compute CLIP embeddings for detected logos
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics import YOLO

# Load model once globally
newspaper_imgs_path = r"C:\Article-Detection-App\NEWSPAPERS_DL\PAGE_IMGS"  # Replace with your folder path
output_folder = r"C:\Article-Detection-App\ARTICLE_IMAGES\articlecrops"  # Folder to save cropped images
article_segemntation_model_path = r"C:\Article-Detection\runs\detect\train\weights\best.pt"
article_segemntation_model_hin_path= r"C:\Article-Detection\runs\detect\Hindi_11_classes\weights\best.pt"
newspaper_segemntation_model_path = r"C:\Article-Detection\runs\detect\train_hin\weights\best.pt"

telugu_newspaper_segemntation_model_path = r"C:\Article-Detection\runs\detect\telegu_eenadu_sakshi_1200_trained\weights\best.pt"
newspaper_segemntation_model_path_for_PS = r"C:\Article-Detection\runs\detect\People_samachar_newspaper_segmentation_4_classes\weights\best.pt" # people Samachar

# logo_detection_model_path = r"C:\Article-Detection\runs\detect\logo_model\best.pt"
logo_detection_model_path = r"C:\Article-Detection\runs\detect\logo_model_v2_trained_on_rotated_image\weights\best.pt"

NOT_MATCHED_LOGO_PATH = r'C:\Article-Detection-App\SCRIPTS\NOT_MATCHED_LOGOS'
ADS_NOT_MAPPED_TO_COMPANY_PATH =  r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY'
LOGO_FOUND_ON_THESE_DISPLAY_PATH = r'C:\Article-Detection-App\SCRIPTS\LOGO_FOUND_ON_THESE_DISPLAY'
# clip_model_path = r""

# previous_newspaper_segemntation_model= r"C:\Article-Detection\runs\detect\train5\weights\best.pt"
# titlemodel_path=r"C:\Article-Detection-App\SCRIPTS\runs\detect\train7\weights\best.pt"
# titleoutput_folder=r"C:\Article-Detection-App\ARTICLE_IMAGES\titlecrops"

article_segemntation_model = YOLO(article_segemntation_model_path)
article_segemntation_model_hin = YOLO(article_segemntation_model_hin_path)
newspaper_segemntation_model = YOLO(newspaper_segemntation_model_path)
telugu_newspaper_segemntation_model = YOLO(telugu_newspaper_segemntation_model_path)
PS_newspaper_segmentation_model = YOLO(newspaper_segemntation_model_path_for_PS)

logo_detection_model = YOLO(logo_detection_model_path)

embedding_model_logo = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(embedding_model_logo)
clip_processor = CLIPProcessor.from_pretrained(embedding_model_logo)

CLASSIFIER_MODEL_PATH = r"C:\Article-Detection\Classifier_models\newspaper_convnext2_best.pt"

#Function to run CLI command and return output
def run_command_with_log(command): 
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)

    combined_output = ""

    while True:
        line = process.stdout.readline()
        if not line:
            break
        combined_output += line
        print(line, end='')

    process.wait()

    return 

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def run_yolo(model, image_path, output_folder, threshold, unq_folder_name):
    print("Threshold for YOLO RUN:", threshold)

    results = model.predict(
        source=image_path,
        conf=threshold,       # same as conf= in CLI
        iou=0.4,              # same as iou=
        save_crop=True,       # save cropped detections
        save_txt=True,        # save label txt files
        save_conf=True,       # include confidences in txt
        project=output_folder, # base output folder
        name=unq_folder_name,  # subfolder name (like CLI's --name)
        exist_ok=True          # overwrite if folder exists
    )

    # Optional: print logs similar to run_command_with_log/
    
    # for r in results:
    #     print(f"✅ Saved results in: {r.save_dir}")
    #     print(f"Detections: {len(r.boxes)}")

    return results


# Load once globally (outside this function)
# article_segmentation_model = YOLO("path/to/article_segmentation_model.pt")

def run_yolo_on_aticle(model, image_path, output_folder):
    """
    Runs YOLO detection on a given image using a preloaded model (no reloading per call).
    Saves cropped detections, txt labels, and confidence values like the CLI version.
    """
    print("Running YOLO on:", image_path)

    results = model.predict(
        source=image_path,
        save_crop=True,      # same as --save_crop
        save_txt=True,       # same as --save_txt
        save_conf=True,      # same as --save_conf
        project=output_folder, # same as --project
        iou=0.2,             # same as --iou=0.2
        exist_ok=True,       # prevents folder errors if re-run
        verbose=True         # print YOLO's internal logs
    )

    # Optional: show saved folder and summary (like your log output)
    # for r in results:
    #     print(f"✅ Saved results in: {r.save_dir}")
    #     print(f"Detected {len(r.boxes)} objects")

    return results

def run_logo_detection_yolo(model, image_path, output_folder):
    """
    Run YOLO model for logo detection.

    Args:
        model: YOLO model object (preloaded YOLO model).
        image_path: Path to the input image or directory of images.
        output_folder: Path to save YOLO output results.
        threshold: Confidence threshold for detections.
        unq_folder_name: Unique folder name for saving outputs.

    Returns:
        results: YOLO detection results object.
    """

    print("🔍 Running YOLO for Logo Detection")
    # print("Threshold for YOLO RUN:", threshold)

    results = model.predict(
        source=image_path,
        #conf=threshold,          # confidence threshold
        iou=0.4,                 # intersection-over-union threshold
        save=True,
        save_crop=True,          # save cropped logo detections
        save_txt=True,           # save label text files
        save_conf=True,          # include confidence scores
        project=output_folder,   # main output directory
        #name=unq_folder_name,    # subdirectory (e.g., unique run name)
        #exist_ok=True            # overwrite if folder already exists
        verbose=True 
    )

    # Optional logging for clarity
    for r in results:
        print(f"✅ Saved logo detection results in: {r.save_dir}")
        print(f"Detected logos: {len(r.boxes)}")

    return results


# ______________Functions to run YOLO model and save cropped images_____________#
def run_yolo_old(model_path, image_path, output_folder, threshold, unq_folder_name):
    print("Threshold for YOLO RUN: ",threshold)
    command = f"yolo task=detect mode=predict model={model_path} source={image_path} conf={threshold} save_crop save_txt save_conf project={output_folder} iou=0.4 name={unq_folder_name}"
    run_command_with_log(command)

def run_yolo_on_aticle_old(model_path, image_path, output_folder):  
    #saving txt file and corresponding confidence so we can pick through highest confidence in case of 2 or more authors , headlines , subheadline , tophead
    command = f"yolo task=detect mode=predict model={model_path} source={image_path} save_crop save_txt save_conf project={output_folder} iou=0.2"
    run_command_with_log(command)

def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding


def get_ollama_prediction(image_path):
    """
    Calls the local Ollama VLM (qwen3-vl:8b) for fallback classification.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return {"label": "ERROR", "confidence": 0}

    prompt = """
        You are an expert newspaper content classifier.

        Classify the newspaper image into ONLY ONE of these 3 categories:
        1. ARTICLE
        2. DISPLAY
        3. CLASSIFIED

        If the image is a puzzle, crossword, sudoku, comic, weather report, share market table, or simply does not fit into the 3 categories above, label it as NONE.

        Definitions:
        ARTICLE: Genuine news reporting, editorial, informative content.
        DISPLAY: Large promotional advertisements, posters, heavy graphic design.
        CLASSIFIED: Small paid advertisements, matrimonial, obituaries, tender notices.

        IMPORTANT RULES:
        - Shradhanjali, obituaries, paid tribute notices are CLASSIFIED.
        - Genuine reporting is ARTICLE.
        - If unsure between DISPLAY and CLASSIFIED, choose CLASSIFIED.

        You MUST evaluate your confidence in this classification on a scale of 0 to 100.
        
        Return the result STRICTLY in the following JSON format:
        {"label": "...", "confidence": ...}
    """

    url = 'http://192.168.3.138:11434/api/generate' 
    
    payload = {
        "model": "qwen3-vl:8b", 
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        # "format": "json",
        "options": {
            "temperature": 0   
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result_text = response.json().get("response", "{}")
        parsed_result = json.loads(result_text)
        
        # Capitalize transforms "ARTICLE" -> "Article", matching your folder structure perfectly
        label = parsed_result.get("label", "NONE").strip().capitalize() 
        confidence = int(parsed_result.get("confidence", 0))
        
        return {"label": label, "confidence": confidence}
        
    except Exception as e:
        print(f"Ollama API/Parsing Error for {os.path.basename(image_path)}: {e}")
        return {"label": "ERROR", "confidence": 0}





#__________________________Summary_________________________________________________#
'''
def summarize_article(article_text, max_length=300, min_length=0):
    # Load tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Specify your model here
    max_input_tokens = tokenizer.model_max_length  # Get the maximum token length for the model

    # Tokenize the input text
    input_tokens = tokenizer.tokenize(article_text)

    # Truncate the article_text if it exceeds the maximum token limit
    if len(input_tokens) > max_input_tokens:
        article_text = tokenizer.convert_tokens_to_string(input_tokens[:max_input_tokens])

    # Ensure the article is long enough to summarize
    if len(article_text.strip()) < min_length:
        return ""  # Return an empty string if the text is too short

    try:
        summarizer = pipeline("summarization", model="t5-small")  # Specify your model here
        summary = summarizer(article_text, max_length=max_length, min_length=min_length, return_text=True)

        # Ensure summary is a list and has the required structure before extracting text
        if isinstance(summary, list) and summary:
            if 'summary_text' in summary[0]:
                return summary[0]['summary_text']  # Return the summary text
            else:
                return ""  # Fallback if structure is unexpected
        else:
            return ""  # Fallback for non-list or empty outputs

    except Exception as e:
        print(f"Error during summarization: {e}")  # Log the error for debugging
        return ""
'''

def preprocess_text(text):
    # Step 1: Remove hyphens between lowercase letters and also remove spaces around it.
    # This will merge words like "Bash-ar" to "Bashar" and "Sund- ay" to "Sunday".
    text = re.sub(r'(?<=\b[a-z])\s*-+\s*(?=[a-z]\b)', '', text)

    # Step 2: Ensure commas in numbers are preserved (e.g., '2,000' remains '2,000')
    text = re.sub(r'(\d),(\d)', r'\1,\2', text)

    # Step 3: Remove unwanted punctuation like colon, semicolon, and dash (except in valid places like numbers).
    # We will leave dashes that are between numbers or in valid contexts.
    text = re.sub(r'[:;,-]', ' ', text)

    # Step 4: Remove any extra spaces to clean up the text
    cleaned_text = ' '.join(text.split())

    return cleaned_text


def word_counter(first_string, second_string):
    if len(first_string) > 0:
        word_count = len(re.findall(r'\s+', first_string)) + 1 if first_string.strip() else 0
    else:
        word_count = len(re.findall(r'\s+', second_string)) + 1 if second_string.strip() else 0
    return word_count
    

def col_counter(width):
    col_count = round(width/3.3)
    if col_count == 0:
        return 1
    return col_count


def summarize_article(text, model_name="facebook/bart-large-cnn", max_length=300, min_length=30):
    try:
        # Preprocess the text
        cleaned_text = preprocess_text(text)

        # Load tokenizer and summarizer for the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model_name)

        # Handle token truncation
        # max_input_tokens = tokenizer.model_max_length
        input_tokens = tokenizer.tokenize(cleaned_text)
        if len(input_tokens) > 1000:
            cleaned_text = tokenizer.convert_tokens_to_string(input_tokens[:1000])

        # Ensure the article is long enough to summarize
        if len(cleaned_text.strip()) < min_length:
            print(f"{cleaned_text.strip()}, min-length is {min_length}")
            return ""
            

        # Generate summary
        summary = summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)

        # Extract summary text
        if isinstance(summary, list) and summary and 'summary_text' in summary[0]:
            return summary[0]['summary_text']
        else:
            print("Unable to extraction")
            return ""

    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""


# def get_db_connection():
#     return mysql.connector.connect(
#         host="103.115.194.67",
#         user="user1",
#         password="Q!23plkmn12ms",
#         database="newspaper"

#     )

def get_db_connection():
    while True:
        try:
            # conn = mysql.connector.connect(
            #     host="localhost",
            #     user="root",
            #     password="4cplus",
            #     database="newspaper",
            #     connection_timeout=60
            # )
            conn = mysql.connector.connect(
                host="103.115.194.67",
                user="user2",
                password="Plm!29Qwsd98c",
                database="newspaper",
                connection_timeout=60
            )
            return conn
        except mysql.connector.Error as err:
            print("Connection failed, retrying...", err)
            time.sleep(5)  # Wait and retry



#_________________________Inserting data into different tables_______________________#
def insert_into_articles(image_name, folder_name, ocr_text, date, keywords, summary, title, newspaper, edition, subedition, page_no, language, sentiment, author,headline, caption, crosser, img_paths, img_withcaption, intro_path, sub_head, tophead, substory_path, story, xml_path, folderpath, top_category_res, authors_email, article_dim, low_res_path, col_count, word_count, slug, headline_word_count, height_article_cm, width_article_cm, area_article_cm_sq, images_inside_article_count, total_image_area_inside_article):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_db_connection()
    cursor = connection.cursor()

    query = "INSERT INTO articles (image_name, folder_name, ocr_text, date, keywords, summary, title, newspaper, edition, subedition, pageno, language, sentiment, author, headtext, caption, crosser, img_paths, img_withcaption, intro_path, sub_head, tophead, substory_path, story, xml_path, folderpath, categorization, authors_email, article_dimension, low_res_path, column_count, words_count, article_slug, headline_word_count, height_article_cm, width_article_cm, area_article_cm_sq, images_inside_article_count, total_image_area_inside_article) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (image_name, folder_name, ocr_text, date, keywords, summary, title, newspaper, edition, subedition, page_no, language, sentiment, author, headline , caption, crosser, img_paths, img_withcaption, intro_path, sub_head, tophead, substory_path, story, xml_path, folderpath, top_category_res, authors_email, article_dim, low_res_path, col_count, word_count, slug, headline_word_count, height_article_cm, width_article_cm, area_article_cm_sq, images_inside_article_count, total_image_area_inside_article)

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()

def insert_into_advertisements(image_name, folder_name, date, newspaper, edition, subedition, page_no, language, adtype, ocr_text_ad, low_res_path, height_Ad_cm, width_Ad_cm, area_Ad_cm_sq, col_count, concatenated_logo_paths, advertising_company, logo_ocr, brand_name_vlm, parent_company_vlm, ad_category_vlm):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_db_connection()
    cursor = connection.cursor()

    query = "INSERT INTO advertisements (image_name, folder_name, date, newspaper, edition, subedition,  pageno, language, adtype, ocr_text, low_res_path, height_Ad_cm, width_Ad_cm, area_Ad_cm_sq, column_count, logos_path, advertising_company, logo_ocr, vlm_Brand, vlm_Parent_Company, vlm_ad_category) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (image_name, folder_name, date, newspaper, edition, subedition, page_no, language, adtype, ocr_text_ad, low_res_path, height_Ad_cm, width_Ad_cm, area_Ad_cm_sq, col_count, concatenated_logo_paths, advertising_company, logo_ocr, brand_name_vlm, parent_company_vlm, ad_category_vlm)

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()

def insert_into_imdeia_advt_dash( date, newspaper, edition, subedition, page_no, adtype, height_Ad_cm, width_Ad_cm, area_Ad_cm_sq, client_name, brand_name, category_name):
    
    connection=get_imedia_database_connection()
    cursor = connection.cursor()

    query = "INSERT INTO advt_dash (PUBLICATION_DATE, PUBLICATION_NAME, EDITION, SUBEDITION,  PAGE_NUMBER, AD_TYPE, HEIGHT, WIDTH, VOLUME, CLIENT_NAME, BRAND_NAME, PRIAD_CATEGORY_NAME) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = ( date, newspaper, edition, subedition, page_no, adtype, height_Ad_cm, width_Ad_cm, area_Ad_cm_sq, client_name, brand_name, category_name)

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()


def insert_into_extras(  folder_name, image_name, date, newspaper,edition, subedition, pageno, language, ocr_text, height_Ex_cm, width_Ex_cm, area_extras_cm_sq, column_count):
  
    connection = get_db_connection()
    cursor = connection.cursor()

    query = """INSERT INTO extras (  folder_name,  image_name,  date, newspaper, edition, subedition, pageno, language, ocr_text, height_Ex_cm,width_Ex_cm,area_extras_cm_sq,  column_count)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (folder_name, image_name,date, newspaper, edition, subedition, pageno, language, ocr_text, height_Ex_cm, width_Ex_cm, area_extras_cm_sq, column_count )

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()


def insert_into_table_pagesummaries(newspaper, edition, subedition, date, pageno, articles, display, classified, totalad, total_article_area_cm_sq, total_Display_Ad_area_cm_sq, total_classified_Ad_area_cm_sq, image_num_per_page, image_area_per_page):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_db_connection()
    cursor = connection.cursor()

    query = "INSERT INTO pagesummaries (newspaper, edition, subedition, date, pageno, articles, display, classified, totalad, total_article_area_cm_sq, total_Display_Ad_area_cm_sq, total_classified_Ad_area_cm_sq, image_num_per_page, image_area_per_page) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (newspaper, edition, subedition, date, pageno, articles, display, classified, totalad, total_article_area_cm_sq, total_Display_Ad_area_cm_sq, total_classified_Ad_area_cm_sq, image_num_per_page, image_area_per_page)

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()
    

#__________________________Deleting data if same newspaper with same edition and same from different tables_____________#
def delete_existing_articles(newspaper, edition, subedition, date, page_no):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_db_connection()
    cursor = connection.cursor()

    print(f"\n Deleting existing articles of\n Newspaper : {newspaper} \n Edition: {edition} \n Subedition:{subedition} \n Date: {date} \n Page Number: {page_no} \n")

    # SQL query to delete existing articles based on 'newspaper', 'date', and 'pageno'
    delete_query = "DELETE FROM articles WHERE newspaper = %s AND edition= %s AND subedition = %s AND date = %s AND pageno = %s"
    delete_values = (newspaper, edition, subedition, date, page_no)
    print(f"\n Delete Query: {delete_query} \n")
    cursor.execute(delete_query, delete_values)
    connection.commit()

    cursor.close()
    connection.close()



def delete_existing_advertisements(newspaper, edition, subedition, date, page_no):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_db_connection()
    cursor = connection.cursor()

    print(f"\n Deleting existing advertisement of\n Newspaper : {newspaper} \n Edition: {edition} \n Subedition:{subedition} \n Date: {date} \n Page Number: {page_no} \n")

    # SQL query to delete existing articles based on 'newspaper', 'date', and 'pageno'
    delete_query = "DELETE FROM advertisements WHERE newspaper = %s AND edition= %s AND subedition = %s AND date = %s AND pageno = %s"
    delete_values = (newspaper, edition, subedition, date, page_no)
    print(f"\n Delete Query: {delete_query} \n")
    cursor.execute(delete_query, delete_values)
    connection.commit()

    cursor.close()
    connection.close()


def get_imedia_database_connection():
    while True:
        try:
            connection = mysql.connector.connect(
                host='103.115.194.43',
                user= 'imedia',
                password='Axcv#@09plm!ntls',
                port=7545,
                database='imedia',
                connection_timeout=60
            )
            return connection
        except mysql.connector.Error as err:
            print('COnnection falied to imedia (Mariadb)...', err)
            time.sleep(5)

    

def delete_advertisements_data_for_advt_dash(newspaper, edition, subedition, date, page_no):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_imedia_database_connection()
    cursor = connection.cursor()

    # SQL query to delete existing articles based on 'newspaper', 'date', and 'pageno'
    delete_query = "DELETE FROM advt_dash WHERE PUBLICATION_NAME = %s AND EDITION= %s AND SUBEDITION = %s AND PUBLICATION_DATE = %s AND PAGE_NUMBER = %s"
    delete_values = (newspaper, edition, subedition, date, page_no)
    print(f"\n Delete Query: {delete_query} \n")
    cursor.execute(delete_query, delete_values)
    connection.commit()

    cursor.close()
    connection.close()

    
def delete_existing_extras(newspaper, edition, subedition, date, page_no):

    connection = get_db_connection()
    cursor = connection.cursor()

    print(
        f"\n Deleting existing extras of\n"
        f" Newspaper : {newspaper}\n"
        f" Edition   : {edition}\n"
        f" Subedition: {subedition}\n"
        f" Date      : {date}\n"
        f" Page No   : {page_no}\n"
    )

    delete_query = """
    DELETE FROM extras WHERE newspaper = %s AND edition = %s AND subedition = %s AND date = %s AND pageno = %s """

    values = (newspaper, edition, subedition, date, page_no)

    cursor.execute(delete_query, values)
    connection.commit()

    cursor.close()
    connection.close()



def delete_existing_pagesummaries(newspaper, edition, subedition, date, page_no):
    # connection = mysql.connector.connect(
    #     host="103.115.194.67",
    #     user="user1",
    #     password="Q!23plkmn12ms",
    #     database="newspaper"

    # )
    connection=get_db_connection()
    cursor = connection.cursor()

    print(f"\n Deleting existing advertisement of\n Newspaper : {newspaper} \n Edition: {edition} \n Subedition:{subedition} \n Date: {date} \n Page Number: {page_no} \n")

    # SQL query to delete existing articles based on 'newspaper', 'date', and 'pageno'
    delete_query = "DELETE FROM pagesummaries WHERE newspaper = %s AND edition= %s  AND subedition = %s AND date = %s AND pageno = %s"
    delete_values = (newspaper, edition, subedition, date, page_no)
    print(f"\n Delete Query: {delete_query} \n")
    cursor.execute(delete_query, delete_values)
    connection.commit()

    cursor.close()
    connection.close()
    


def delete_images_in_folder(folder_path):    
    for item in os.listdir(folder_path): 
        item_path = os.path.join(folder_path, item)

        # Check if the item has an image extension
        if item.lower().endswith(('.png', '.jpg', '.jpeg')):      # '.gif', '.bmp', '.tiff', '.webp'
            os.unlink(item_path)  # Delete the image file
    
def extract_info_from_filename(file_name): # Extract the publication, edition, date and page number from the file name
        parts = file_name.split('_')
        print(parts)
        newspaper = parts[0]
        edition = parts[1]
        subedition = parts[2]
        date = datetime.strptime(parts[3], "%d%m%y").strftime("%Y-%m-%d")
        language = parts[4]
        page_no = parts[5]
        return newspaper, edition, subedition, date, language, page_no

def folder_exists(directory_path, folder_name):
    """
    Check if a folder exists in a directory.
    
    Args:
    - directory_path (str): Path to the directory where you want to check.
    - folder_name (str): Name of the folder you're checking for.

    Returns:
    - bool: True if folder exists, False otherwise.
    """
    # Create the full path to the folder
    folder_path_join = os.path.join(directory_path, folder_name)
    print(os.path.exists(folder_path_join) and os.path.isdir(folder_path_join))
    # Check if the folder path exists and is a directory, return true or false
    return os.path.exists(folder_path_join) and os.path.isdir(folder_path_join)

def move_imgs_to_done(source_folder, done_folder, file_name):
    """Move all PDF files from the source folder to the 'Done' subfolder."""
    # Create the 'Done' subfolder if it doesn't exist
    os.makedirs(done_folder, exist_ok=True)

    # for file_name in os.listdir(source_folder):
    source_path = os.path.join(source_folder, file_name)

        # Check if the file is a PDF
    # if file_name.lower().endswith((".png", ".jpg")):
    destination_path= os.path.join(source_folder,done_folder)
    destination_path = os.path.join(destination_path, file_name)
    print("Destination Folder: ", destination_path)
    shutil.move(source_path, destination_path)
    print(f"moved images from {source_folder} to done folder {destination_path}")

def create_folder(path):
    # Create the directory and any necessary parent directories
    os.makedirs(path, exist_ok=True)

def delete_folder(folder_path):
    shutil.rmtree(folder_path)

def add_text_to_image(image_path, text, font_path=None):
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Set a base font size
    base_font_size = int((img_height + img_width) * 0.02)  
    if font_path:
        font = ImageFont.truetype(font_path, base_font_size)
    else:
        font = ImageFont.load_default()

    # Wrap text to fit within image width
    max_chars_per_line = img_width // (base_font_size // 2)
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)
    
    # Count the number of lines needed
    text_lines = wrapped_text.split("\n")
    num_lines = len(text_lines)
    
    # Dynamically set white space height
    line_spacing = int(base_font_size * 1.2)  
    white_space_height = num_lines * line_spacing + 10  

    # Create a new image with extra space
    new_img = Image.new("RGB", (img_width, img_height + white_space_height), "white")
    
    # Paste the original image on top
    new_img.paste(img, (0, 0))
    
    # Get a drawing context
    draw = ImageDraw.Draw(new_img)

    # Set text starting position (centered vertically)
    text_y = img_height + (white_space_height - (num_lines * line_spacing)) // 2
    
    # Draw each line, centering it horizontally
    for line in text_lines:
        # text_width, _ = draw.textsize(line, font=font)  # Get text width
        text_width = draw.textbbox((0, 0), line, font=font)[2]
        text_x = (img_width - text_width) // 2  # Center horizontally
        draw.text((text_x, text_y), line, fill="black", font=font)
        text_y += line_spacing  # Move down for the next line
    
    # Save the final image
    new_img.save(image_path)


def generate_slug(article_image_name):
    slug = article_image_name + "-" + str(random.randint(0, 1000)) + "-" + str(random.randint(2000, 3000))
    return slug


def copy_images(source_path, destination_path):
    try:
        shutil.copy2(source_path, destination_path)
    except FileNotFoundError:
        print(f'\nError: File Not Found Error {source_path}')
    except PermissionError:
        print(f'Error: Permission Error for the file {source_path}')
    except Exception as e:
        print(f'Error in copying: {e}')



def send_low_conf_images_to_ollama(folder_to_check, prompt):

    article_folder = os.path.join(folder_to_check, "Article")

    classified_folder = os.path.join(folder_to_check, "Classified")
    display_folder = os.path.join(folder_to_check, "Display")
    extras_folder = os.path.join(folder_to_check, "Extras")

    if not os.path.exists(article_folder):
        print("Article folder not found")
        return
    for img in os.listdir(article_folder):
        img_path = os.path.join(article_folder, img)
        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(
                f.read()
            ).decode("utf-8")

        payload = {
            "model": "qwen3-vl:8b",
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0
            }
        }
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        result = response.json().get(
            "response",
            ""
        ).strip().upper()

        print(f"{img} => {result}")
        if result == "CLASSIFIED":
            shutil.move(
                img_path,
                os.path.join(classified_folder, img)
            )
        elif result == "DISPLAY":
            shutil.move(
                img_path,
                os.path.join(display_folder, img)
            )
        elif result == "EXTRAS":
            shutil.move(
                img_path,
                os.path.join(extras_folder, img)
            )

def get_low_confidence_crops(
    label_file_path,
    crops_root,
    class_names,
    threshold=0.75
):
    
    '''
    we require this function becacause yolo was mixing classes of the images, 
    so we will find images with low confidence than 75% or 80% then send them to classifier (trained on efficient_net_b0),
    if it also gives low confidence then we will send them to ollama, running on 2 systems(192.168.4.138, 192.168.4.61)
    '''

    """
    Parameters
    ----------
    label_file_path : str
        Path to YOLO txt label file

    crops_root : str
        Path to crops folder

    class_names : dict or list
        Example:
        {0: "Article", 1: "Classified", 2: "Display", 3: "Extras"}

    threshold : float
        Confidence threshold

    Returns
    -------
    list[dict]
    """

    label_file_path = Path(label_file_path)
    crops_root = Path(crops_root)

    image_base_name = label_file_path.stem

    low_conf_items = []

    # Track occurrence count for each class
    class_occurrence_counter = {}

    with open(label_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, start=1):

        parts = line.strip().split()

        if len(parts) < 6:
            continue

        class_id = int(parts[0])
        confidence = float(parts[-1])

        # Update occurrence count
        class_occurrence_counter[class_id] = (
            class_occurrence_counter.get(class_id, 0) + 1
        )

        occurrence = class_occurrence_counter[class_id]

        # Only interested in low confidence detections
        if confidence >= threshold:
            continue

        class_name = (
            class_names[class_id]
            if isinstance(class_names, dict)
            else class_names[class_id]
        )

        # YOLO crop naming logic
        if occurrence == 1:
            crop_filename = f"{image_base_name}.jpg"
        else:
            crop_filename = f"{image_base_name}{occurrence}.jpg"

        crop_path = crops_root / class_name / crop_filename

        low_conf_items.append(
            {
                "line_no": line_no,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "occurrence": occurrence,
                "crop_path": str(crop_path),
            }
        )

    return low_conf_items

def log_correction(destination_folder, filename, confidence, source):
    """
    Creates or appends to a txt file logging the corrections made by Ollama or Classifier.
    """
    log_file = os.path.join(destination_folder, "corrections_log.txt")
    index = 0
    
    # Calculate index by checking existing lines
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            index = len([line for line in f.readlines() if line.strip()])
            
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{index}, confidence of {filename} by {source} is {confidence}\n")

def move_corrected_image(source_path, destination_folder, old_class):
    """
    Cuts an image from its current path and moves it to the exact destination folder.
    If a file collision occurs, appends a '_from_ClassNames' suffix to prevent overwriting.
    """
    try:
        # Make sure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)
        
        # Extract the current filename (e.g., "page1.jpg")
        filename = os.path.basename(source_path)
        destination_path = os.path.join(destination_folder, filename)
        
        # Check if a file with the exact same name already lives in the destination
        #if os.path.exists(destination_path):
        # Split filename into name and extension (e.g., "page1" and ".jpg")
        base, ext = os.path.splitext(filename)
        
        # Make sure it matches your plural style (e.g., "Article" -> "Articles")
        folder_suffix = f"{old_class}s" if not old_class.lower().endswith('s') else old_class
        
        # Rebuild the file name with the suffix (e.g., "page1_from_Displays.jpg")
        filename = f"{base}_from_{folder_suffix}{ext}"
        destination_path = os.path.join(destination_folder, filename)
        print(f"   [COLLISION DETECTED] Renaming to '{filename}' to avoid overwrite.")
        
        # Perform the actual cut and paste
        shutil.move(source_path, destination_path)
        print(f"   [SUCCESS] Moved '{filename}' to the '{os.path.basename(destination_folder)}' folder.")
        
        # Return the final path (complete with the new filename if it changed)
        return destination_path
        
    except Exception as e:
        print(f"   [ERROR] Failed to move file {source_path}: {e}")
        return None

# Main function
def main(newspaper_img):

    try:

        '''When we performe YOLO prediction on images it would create folders like predict10, predict11, predict12 ...
        this will keep incrementing if we didn't pass NAME parameter, in name parameter we can provide name of the folder we want.
        so previously prediction folder was incrementing by 1 each time and we were keeping track of it in .ENV file in ADMIN folder
        but now we are passing the name parameter.'''
        # i = envc.get_env_value(key='ARTICLEFOLDER_LAST', env_path="C:\\Article-Detection-App\\ADMIN\\.env")
        # j = envc.get_env_value(key='TITLEFOLDER_LAST', env_path="C:\\Article-Detection-App\\ADMIN\\.env")

        
        unq_folder_name = newspaper_img.split('.')[0]
        newspaper, edition,subedition, date, language, page_no = extract_info_from_filename(newspaper_img)
        delete_existing_articles(newspaper,edition,subedition,date,page_no)
        delete_existing_advertisements(newspaper,edition,subedition,date,page_no)
        # delete_advertisements_data_for_advt_dash(newspaper,edition,subedition,date,page_no)
        delete_existing_extras(newspaper,edition,subedition,date,page_no)
        delete_existing_pagesummaries(newspaper,edition,subedition,date,page_no)
        image_path = os.path.join(newspaper_imgs_path, newspaper_img)
        
        #Deleting if folder exists(Incase we are uploading, same newspaper, same edition and same date AGAIN)
        if folder_exists(output_folder, unq_folder_name):
            print("Deleting Folder:", unq_folder_name)
            delete_folder(os.path.join(output_folder,unq_folder_name))

        print("----------------------Starting YOLO on Newspaper Image.------------------")
        if language == 'eng' or language == '':
            print("YOLO FOR ARTICLE STARTING")
            run_yolo(newspaper_segemntation_model, image_path, output_folder,0.60, unq_folder_name)
            print("YOLO FOR ARTICLE ENDED")
        
        elif language == 'hin':
            if newspaper == 'PS':
                print("YOLO FOR hindi PS ARTICLE STARTING")
                run_yolo(PS_newspaper_segmentation_model, image_path, output_folder,0.40, unq_folder_name)
            else:
                print("YOLO FOR hindi ARTICLE STARTING")
                run_yolo(newspaper_segemntation_model, image_path, output_folder,0.40, unq_folder_name)
                print("YOLO FOR ARTICLE ENDED")
        
        elif language == 'mar':
            print("YOLO FOR marathi ARTICLE STARTING")
            run_yolo(newspaper_segemntation_model, image_path, output_folder,0.40, unq_folder_name)
            print("YOLO FOR ARTICLE ENDED")
        elif language=="tel":
            print("YOLO FOR telegu ARTICLE STARTING")
            run_yolo(telugu_newspaper_segemntation_model, image_path, output_folder,0.40, unq_folder_name)
            print("YOLO FOR ARTICLE ENDED")
            
        else:
            print("YOLO IN ELSE CONDITION ARTICLE STARTING")
            run_yolo(newspaper_segemntation_model, image_path, output_folder,0.40, unq_folder_name)
        #Incrementing i varible with new predictions (not now)
        # i= int(i)
        # i+=1
        # envc.update_env_variable(key= 'ARTICLEFOLDER_LAST',value= i , env_path="C:\\Article-Detection-App\\ADMIN\\.env")
        # if i==1:
        #     print('if',i)
        #     predict_path=f'predict\crops'
        # else:
        #     print('else',i)
        #     predict_path=f'predict{i}\crops'


        predict_path = f'{unq_folder_name}\crops'
        folder_to_check = os.path.join(output_folder,predict_path)
        print(folder_to_check)
        
        # Go to txt file and find any file cropping that has confidence less than 80 or 75 percent (5th index or 6th position)
        # label_folder_path = f'{unq_folder_name}\\labels'
        label_txt_path = os.path.join(output_folder, f'{unq_folder_name}\\labels\\{unq_folder_name}.txt' )

        
        class_names = {
            0: "Article",
            1: "Classified",
            2: "Display",
            3: "Extras",
            4: "Feature",
            5: "Header"
        }


        # Initialize the classifier safely inside the process
        classifier = NewspaperCropClassifier(model_path=CLASSIFIER_MODEL_PATH)

        items = get_low_confidence_crops(
            label_file_path=label_txt_path,
            crops_root=folder_to_check,
            class_names=class_names,
            threshold=0.90
        )

        print(f"\n{'IMAGE PATH':<70} | {'OLD PREDICTION':<20} | {'NEW PREDICTION'}")
        print("-" * 120)

        for item in items:
            class_id_of_low_conf_image = item["class_id"]
            
            if class_id_of_low_conf_image != 3:              # not for Extras folder right now
                crop_path = item["crop_path"]

                if not Path(crop_path).exists():
                    print(f"Missing crop: {crop_path}")
                    continue

                #Pass To Classifier
                classifier_result = classifier.predict(crop_path)

                if classifier_result:
                    old_pred = f"{item['class_name']} ({item['confidence']:.2f})"
                    new_pred = f"{classifier_result['predicted_class']} ({classifier_result['confidence']}%)"
                    
                    print(f"{crop_path:<70} | {old_pred:<20} | {new_pred}")

                    if classifier_result['confidence'] < 85:
                        print("Calling Ollama")
                        old_class = item["class_name"]
                        
                        # Fetch new class dynamically from the VLM
                        ollama_result = get_ollama_prediction(crop_path)
                        new_class = ollama_result["label"]
                        print(f"Ollama prediction: {new_class} ({ollama_result['confidence']}%)")

                        # Handle Ollama fallback
                        valid_ollama_classes = ["Article", "Classified", "Display"]
                        if new_class not in valid_ollama_classes:
                            print(f"Ollama returned '{new_class}'. Leaving image as is.")
                            continue

                        #Olama will predict from the image, lets say it is looking in article and he founds classified, it moves it to classified (like we did in classifier logic, olama will also use the same move_corrected_image)
                        
                        if old_class == new_class:
                            print(f"Image is correctly placed.")
                            continue
                            
                       # if YOLO is wrong and high confidence, listen to classifier (now Ollama)
                        else:
                            print(f"Moving from '{old_class}' to '{new_class}'...")
                            destination_folder = os.path.join(folder_to_check, new_class)
                            # MOVE
                            new_crop_path = move_corrected_image(crop_path, destination_folder, f"frm_{old_class[:4]}")
                            
                            # Update the dictionary so any downstream logic knows the new location
                            if new_crop_path:
                                item["crop_path"] = new_crop_path
                                item["class_name"] = new_class
                                log_correction(destination_folder, os.path.basename(new_crop_path), ollama_result['confidence'], "ollama")
                        
                    else:
                        old_class = item["class_name"]
                        new_class = classifier_result["predicted_class"]
                        
                        # Is the image already in the correct folder?
                        if old_class == new_class:
                            print(f"Image is correctly placed.")
                            continue
                            
                        # if YOLO is wrong and high confidence, listen to classifier
                        else:
                            print(f"Moving from '{old_class}' to '{new_class}'...")
                            destination_folder = os.path.join(folder_to_check, new_class)
                            # MOVE
                            new_crop_path = move_corrected_image(crop_path, destination_folder, f"frm_{old_class[:4]}")
                            
                            # Update the dictionary so any downstream logic knows the new location
                            if new_crop_path:
                                item["crop_path"] = new_crop_path
                                item["class_name"] = new_class
                                log_correction(destination_folder, os.path.basename(new_crop_path), classifier_result['confidence'], "classifier")
                                  
        
        
        # for Dimension of the newspapere
        newpaper_dimension_centimeters = {'HT': [56, 41], 'TOI': [56, 41], 'LOK': [51, 33]}
        article_in_pixel = [51, 33]  # Default value
        for key, values in newpaper_dimension_centimeters.items():
            if newspaper == key:
                article_in_pixel = values
                break  # Exit the loop once a match is found   

        article_num = 0
        total_article_area_cm_sq = 0
        image_num_per_page = 0
        image_area_per_page = 0
        if folder_exists(folder_to_check,'Article'):
            print(folder_exists)
            print("ARTICLES FOUND")
            predict_path=os.path.join(predict_path,'Article')
            output_path= os.path.join(output_folder, predict_path)

            low_resl_folder = os.path.join(output_path, "low_resl_imgs")
            create_folder(low_resl_folder)

            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

            # List all files in the directory and filter by image extensions
            image_files = [f for f in os.listdir(output_path) if f.lower().endswith(image_extensions)]

            for imagefilename in image_files:
                images_DB = []  # for inserting data of images inside articles into images table -- foreign key SLUG of article (slug is unique in articles -- obtained on the basis of imagename of the article)
                images_inside_article_count = 0
                total_image_area_inside_article = 0
                slug = generate_slug(imagefilename)

                low_res_path = "low_resl_imgs-" + imagefilename
                article_num += 1
                print(f"\n LANGAUGE IS {language} \n")
                cropped_path= os.path.join(output_path, imagefilename)
                print(f"PATH OF INDIVIDUAL CROPPED ARTICLE: \n {cropped_path} \n")

                #_____________Creating low resolution images to show on thumbnail on website_______________________#
                create_low_res(output_path,imagefilename, low_resl_folder)

                
                #------------------------------------------------------------------------------------------------------------------------------------------------
                # nc: 11                                                                                                                                        -
                # names: ['author', 'caption', 'crosser', 'headline', 'image', 'image_and_caption', 'intro', 'story', 'subheadline', 'tophead', 'u_substory']   -   
                #                                                                                                                                               -
                #                                                                                                                                               -
                # 'author':             0,                                                                                                                      -                                                                                                                      -
                # 'caption':            1,                                                                                                                      -
                # 'crosser':            2,                                                                                                                      - 
                # 'headline'            3,                                                                                                                      - 
                # 'image'               4,                                                                                                                      - 
                # 'image_and_caption'   5,                                                                                                                      - 
                # 'intro'               6,                                                                                                                      - 
                # 'story'               7,                                                                                                                      - 
                # 'subheadline'         8,                                                                                                                      - 
                # 'tophead'             9,                                                                                                                      - 
                # 'u_substory'          10                                                                                                                      -
                #                                                                                                                                               -
                #------------------------------------------------------------------------------------------------------------------------------------------------

                

                # Running yolo model on each article for extracting different parts of article (across 11 clases) after articles have been cropped from the newspaper page
                
                #________________________________Yolo Model for article segmentation (START)_______________________________________________________________________________

                if imagefilename.endswith('.jpg') or imagefilename.endswith('.png') or imagefilename.endswith('.jpeg'):
                    folder_name_from_image = imagefilename.split('.')[0]
                print(folder_name_from_image)
                output_folder_article_segmentation = os.path.join(output_path, folder_name_from_image)
                print("Running Yolo on ",output_folder_article_segmentation)
                if language=="eng":
                    run_yolo_on_aticle(article_segemntation_model, cropped_path, output_folder_article_segmentation)
                    print("Completed Yolo on article segmentation(eng)")
                elif language=="hin":
                    run_yolo_on_aticle(article_segemntation_model_hin, cropped_path, output_folder_article_segmentation)
                    print("Completed Yolo on article segmentation(hindi)")
                
                elif language=="mar":
                    run_yolo_on_aticle(article_segemntation_model_hin, cropped_path, output_folder_article_segmentation)
                    print("Completed Yolo on article segmentation(marathi)")

                elif language=="tel":
                    run_yolo_on_aticle(article_segemntation_model_hin, cropped_path, output_folder_article_segmentation)
                    print("Completed Yolo on article segmentation(telegu)")
                else:
                    run_yolo_on_aticle(article_segemntation_model_hin, cropped_path, output_folder_article_segmentation)
                    print(f"Completed Yolo on article segmentation {language}")


                output_folder_article_segmentation_crops = os.path.join(output_folder_article_segmentation, 'predict\crops')
                output_folder_article_segmentation_labels = os.path.join(output_folder_article_segmentation, 'predict\labels')


                #_______________________________________________________________getting different section of article in either ocr or concatinating the imases names(start)___________________________________________________

                author = ''
                headline = ''
                subheadline = ''
                tophead = ''

                dict_for_unique_classes = {"author": 0,
                                        "headline": 3,
                                        "subheadline": 8,
                                        "tophead": 9}
                

                for key , values in dict_for_unique_classes.items():
                    crop_path = os.path.join(output_folder_article_segmentation_crops, key)
                    label_path = os.path.join(output_folder_article_segmentation_labels, f'{folder_name_from_image}.txt')
                    output_image_path = os.path.join(crop_path, f'{folder_name_from_image}.jpg')

                    if os.path.exists(crop_path) and os.path.isdir(crop_path):
                        if len(os.listdir(crop_path)) > 1:          

                            delete_images_in_folder(crop_path)
                            print(f'Deleting images from {crop_path}')
                            high_conf_coordinates = compare_conf(label_path, values, 5) # 5 because confidence comes at 5th position <class_id> , <center_x> <center_y> <height> <width> <conf>
                            print(f'Highest confidence coordinates {high_conf_coordinates}')
                            generate_cropped_image(cropped_path, high_conf_coordinates,output_image_path)
                            print("Generated masked image")
                            ocr_result = do_ocr(output_image_path, language)
                            print(f'{key} : {ocr_result}')
                            
                            if key == "author":
                                author = ocr_result
                            elif key == "headline":
                                headline = ocr_result
                            elif key == "subheadline":
                                subheadline = ocr_result
                            elif key == "tophead":
                                tophead = ocr_result
                        else:
                            ocr_result = do_ocr(output_image_path, language)
                            print(f'ORIGINAL   {key} : {ocr_result}')
                            if key == "author":
                                author = ocr_result
                            elif key == "headline":
                                headline = ocr_result
                            elif key == "subheadline":
                                subheadline = ocr_result
                            elif key == "tophead":
                                tophead = ocr_result
                

                authors_email = ''
                if author != '':
                    author, authors_email = separate_email_from_text(author)
                
                story_text = ''
                story_folder = os.path.join(output_folder_article_segmentation_crops,'story')
                if os.path.exists(story_folder) and os.path.isdir(story_folder):
                    if len(os.listdir(story_folder)) > 1:
                        saving_mask_path = os.path.join(story_folder, f'{folder_name_from_image}_masked_story.jpg')
                        print(f"Mask Image saving path :{saving_mask_path}")

                        masked_image(cropped_path, os.path.join(output_folder_article_segmentation_labels, f'{folder_name_from_image}.txt'), 7,saving_mask_path) ## 7- for story
                        print("Masked image created")
                        
                        story_text = do_ocr(saving_mask_path, language)
                        print(f"Story_text  : {story_text}")        

                    else:
                        path_of_single_story = os.path.join(story_folder, f'{folder_name_from_image}.jpg')
                        story_text = do_ocr(path_of_single_story, language)
                        print(f"Story_text_original  : {story_text}")


                #_________________________________________________Concatination of paths of image classes(start)__________________________________________________________________________________________________
                crosser_paths = ''
                image_paths = ''
                image_and_caption_paths = ''
                caption_paths = ''
                intro_paths = ''
                substory_paths = ''

                keys = ["crosser", "image", "image_and_caption", "caption", "intro", "u_substory"]

                for key in keys:
                    crop_path = os.path.join(output_folder_article_segmentation_crops, key)

                    # Check if the folder exists and is a directory
                    if os.path.exists(crop_path) and os.path.isdir(crop_path):
                        filenames = os.listdir(crop_path)

                        concatenated_paths = "/".join(filenames)
                        print(f"{key} :: {concatenated_paths}")

                        # Update the corresponding variable
                        if key == "crosser":
                            crosser_paths = concatenated_paths
                            
                        elif key == "image":
                            
                            image_paths = concatenated_paths
                            for images_inside_article in os.listdir(crop_path):
                                height_image_inside_article_cm, width_image_inside_article_cm, area_image_inside_article_cm_sq = dimension_acc_newspaper(image_path, os.path.join(crop_path, images_inside_article),  article_in_pixel)


                                # Store related images 
                                # {"image_name": images_inside_article, "width": width_article_cm, "height": height_article_cm, "slug": article["slug"]},

                                images_DB.append(
                                    {"image_name": images_inside_article, "width": width_image_inside_article_cm, "height": height_image_inside_article_cm, "Area":area_image_inside_article_cm_sq, "slug": slug},
                                )

                                images_inside_article_count += 1
                                total_image_area_inside_article += area_image_inside_article_cm_sq
                                image_num_per_page +=1
                                image_area_per_page += area_image_inside_article_cm_sq

                                
                        elif key == "image_and_caption":
                            image_and_caption_paths = concatenated_paths
                        elif key == "caption":
                            caption_paths = concatenated_paths
                        elif key == "intro":
                            intro_paths = concatenated_paths
                        elif key == "u_substory":
                            substory_paths = concatenated_paths

                        print(key)
                #__________________________________________Concatination of paths of image classes(eND)__________________________________________________________________________________________________
                #_________________________________________getting different section of article in either ocr or concatinating the imases names(END)___________________________________________________



                ## ___________________________________________XML(start)_____________________________________________________________________________________________________________________________
                # Make a dictionary which has below key and value for each labels
                print("Starting Xml File generation")
                labels_info=[
                    {
                    "label":"tophead",
                    "label_id":"id",
                    "label_val": "1",
                    "content":tophead
                },
                    {
                    "label":"headline",
                    "label_id":"id",
                    "label_val": "1",
                    "content":headline
                },
                {
                    "label":"subheadline",
                    "label_id":"id",
                    "label_val": "1",
                    "content":subheadline
                },
                {
                    "label":"author",
                    "label_id":"id",
                    "label_val": "1",
                    "content":author
                },
                {
                    "label":"story",
                    "label_id":"id",
                    "label_val": "1",
                    "content":story_text
                },
                {
                    "label":"image",
                    "label_id":"id",
                    "label_val": "1",
                    "content":image_paths
                },
                {
                    "label":"image_and_caption",
                    "label_id":"id",
                    "label_val": "1",
                    "content":image_and_caption_paths
                },
                {
                    "label":"caption",
                    "label_id":"id",
                    "label_val": "1",
                    "content":caption_paths
                },
                {
                    "label":"crosser",
                    "label_id":"id",
                    "label_val": "1",
                    "content":crosser_paths
                },
                {
                    "label":"intro",
                    "label_id":"id",
                    "label_val": "1",
                    "content":intro_paths
                },
                {
                    "label":"substory",
                    "label_id":"id",
                    "label_val": "1",
                    "content":substory_paths
                }
                ]
                data=ET.Element('Article')

                for article in labels_info:
                    content_ext=article.get("content")
                    if article.get("content").strip()!="":
                        temp=article.get("label")
                        var1=temp+"_tag"
                        tag=temp+"tag"
                        globals()[var1]=ET.SubElement(data,tag)
                        var2=temp+"id"
                        globals()[var2]=article.get("label_id")
                        var3=temp+"val"
                        globals()[var3]=article.get("label_val")
                        globals()[var1].set(globals()[var2],globals()[var3])
                        globals()[var1].text=content_ext

                xml_str = ET.tostring(data, encoding="utf-8")

                pretty_xml= parseString(xml_str).toprettyxml(indent=" ")

                xml_path = os.path.join(output_folder_article_segmentation,f'{folder_name_from_image}.xml')
                # GFG77.xml will be replaced with xml file name required
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(pretty_xml)
                print("Xml File Generated")

                

                ##_________________________________XML(End)___________________________________________________________________________________________________________


                
                print("OCR starting on article cropped image... \n")
                soup = do_hocr(cropped_path, language)

                ocr_text = extract_story_from_soup(soup)      
                print(f"\n\n OCR TEXT {ocr_text}\n\n")
                print("OCR ended for article cropped image.\n")
                # Get other required data
                image_name = os.path.basename(cropped_path)
                print("ACTUAL PATH OF IMAGE FOLDER (predict_path):  ",predict_path)
                output_string = predict_path.replace('\\', '-')
                print("PATH after replacing \ with - to input in database:  ",output_string)
                folder_name = output_string
                print("DATE BEING INPUT INTO DATABASE:  ", date)
                if(language=='eng'):
                    summary= summarize_article(ocr_text)
                    keywords= extract_keywords(ocr_text,'en',  num_keywords=5)
                    keywords_str = ','.join(keywords)
                else:
                    
                    if (language=='hin'):
                        summary = mbart.summarize_multilingual_article(ocr_text)
                        keywords= extract_keywords(ocr_text,'hi', num_keywords=5)            
                    elif (language== 'mar'):
                        if story_text != '':
                            summary = mbart.summarize_multilingual_article(story_text)
                        else:
                            summary = mbart.summarize_multilingual_article(ocr_text)
                        keywords= extract_keywords(ocr_text,'mr', num_keywords=5)
                    else:
                        summary = mbart.summarize_multilingual_article(ocr_text)
                        keywords= extract_keywords(ocr_text,'hi', num_keywords=5)
                    keywords_str = ','.join(keywords)
                print("SUMMARY OF ARTICLE :   ",summary)

                
                

                # print("Title Model: ",titlemodel_path)
                # titleimage_path=cropped_path
                # print("article image path = title image input path = ",titleimage_path)
                # print("YOLO FOR TITLE DETECTION STARTING")
                # run_yolo_title(titlemodel_path, titleimage_path, titleoutput_folder, 0.95)
                # print("YOLO FOR TITLE DETECTION ENDED")
                # j= int(j)
                # j+=1
                # envc.update_env_variable(key= 'TITLEFOLDER_LAST',value= j , env_path="C:\\Article-Detection-App\\ADMIN\\.env")
                # if j==1:
                #     title_predict=f'predict\crops'
                # else:
                #     title_predict=f'predict{j}\crops'
                # crop_title= os.path.join(titleoutput_folder,title_predict)
                # print("title crops folder path:  ", crop_title)
                # if folder_exists(crop_title,'Title'):
                #     crop_title= os.path.join(crop_title,'Title')
                #     print("ffffffffffinal path of title image folder ",crop_title)
                #     for titleimgname in os.listdir(crop_title):
                #         croppedtitle_path= os.path.join(crop_title,titleimgname)
                #         print("found title image with path :  ", croppedtitle_path)
                #         title = do_ocr(croppedtitle_path, language)
                #         print("output from title ocr:  ", title)
                # else:
                #     title= tle.extract_headline(soup)
                #     print("title from automated function because title folder not found in crops: ", title)
                
                # if not title:
                title = ''
                if headline.strip() == '':
                
                    title= tle.extract_headline(soup)
                    print("Using title from first line function because title text from ocr is empty: ", title)

                    if not title:
                        title = headline

                

                if (language== 'mar'):
                    # if story_text != '':
                    #     sentiment = get_marathi_sentiment(story_text)
                    # else :
                    #     sentiment = get_marathi_sentiment(ocr_text)
                    sentiment = 0.5
                else: 
                    sentiment = get_sentiment(ocr_text)
                print("sentiment")
                print(f"\n\n {sentiment}\n\n")


    ## _______________________________categorization______________________________________________________________ 

                top_category_res = ''
                if title != '' and language in ("eng"):
                    title_len= len(title.split())
                    check=False
                    if title_len>5:
                        top_category_res,check=categorization(title)
                        if check and summary != '':
                            top_category_res,check=categorization(summary)

                    elif len(summary.split())>25:
                        top_category_res,check = categorization(summary)

                    else:
                        print("Categorization is not possible for the given article.")


    ## _______________________________categorization(END)______________________________________________________________


    #_________________________________dimension of article in centimeters_____________________________________________                      
                # newpaper_dimension_centimeters = {'HT': [56, 41], 'TOI': [56, 41], 'LOK': [51, 33]}

                # article_in_pixel = [51, 33]  # Default value

                # for key, values in newpaper_dimension_centimeters.items():
                #     if newspaper == key:
                #         article_in_pixel = values
                #         break  # Exit the loop once a match is found        
                
                
                height_article_cm, width_article_cm, area_article_cm_sq = dimension_acc_newspaper(image_path, cropped_path, article_in_pixel)
                article_dim = str(round(height_article_cm)) + "~" + str(round(width_article_cm)) + "~" + str(round(area_article_cm_sq))
                print("Article Dimension: ",article_dim)
                
                total_article_area_cm_sq += area_article_cm_sq



    # ____________________________________________column count_______________________
                col_count = col_counter(width_article_cm)

    # _______________________________________________word count__________________
                word_count = word_counter(story_text, ocr_text)

    # _______________________________________________HEadline word count__________________
                headline_word_count = word_counter(headline, title)

    # ________________________________text below image____________________

                # Example usage
                text = f"{newspaper} {edition} {subedition} {date}, H-{round(height_article_cm)} cm, W-{round(width_article_cm)} cm, A-{round(area_article_cm_sq)} sq.cm, Pg.-{page_no}"
                add_text_to_image(os.path.join(output_path, imagefilename), text, font_path="arial.ttf")


                # Insert data into SQL database
                insert_into_articles(image_name, folder_name, ocr_text, date, keywords_str, summary, title, newspaper, edition, subedition, page_no, language, sentiment, author, headline, caption_paths, crosser_paths, image_paths, image_and_caption_paths, intro_paths, subheadline, tophead, substory_paths, story_text, xml_path, folder_name_from_image, top_category_res, authors_email, article_dim, low_res_path, col_count, word_count, slug, headline_word_count, height_article_cm, width_article_cm, area_article_cm_sq, images_inside_article_count, total_image_area_inside_article)

                # article_slug VARCHAR(255),
                # image_path VARCHAR(255),
                # width INT,
                # height INT,
                # AREA INT,

                # images_DB.append([
                #                     {"image_name": images_inside_article, "width": width_image_inside_article_cm, "height": height_image_inside_article_cm, "Area":area_image_inside_article_cm_sq, "slug": slug},
                #                 ])

                if len(images_DB) > 0:
                    image_query = """
                        INSERT INTO images (article_slug, image_path, width, height, area)
                        VALUES (%s, %s, %s, %s, %s)
                    """

                
                    connection=get_db_connection()
                    cursor = connection.cursor()

                    image_data = [(img["slug"], img["image_name"], img["width"], img["height"], img["Area"]) for img in images_DB]
                    cursor.executemany(image_query, image_data)

                    connection.commit()

                    cursor.close()
                    connection.close()

        else:
            print(folder_exists)
            print("No Articles FOUND")
        
        display_ad_num = 0
        total_Display_Ad_area_cm_sq = 0
        if os.path.exists(os.path.join(folder_to_check, 'Display')):
            #go to predict 
            print(folder_exists)
            print("Display Advertisement FOUND")
            predict_path=f'{unq_folder_name}\crops'
            predict_path=os.path.join(predict_path,'Display')
            folder_name = predict_path.replace('\\', '-')

            output_path= os.path.join(output_folder, predict_path)

            low_resl_folder = os.path.join(output_path, "low_resl_imgs")
            create_folder(low_resl_folder)

            adtype = 'Display'
            # Extract cropped images and perform OCR
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

            # List all files in the directory and filter by image extensions
            image_files = [f for f in os.listdir(output_path) if f.lower().endswith(image_extensions)]
            for imagefilename in image_files:
                low_res_path = "low_resl_imgs-" + imagefilename
                display_ad_num += 1
                cropped_path= os.path.join(output_path, imagefilename)
                print(f"PATH OF INDIVIDUAL CROPPED Advertisement: \n {cropped_path} \n")
                create_low_res(output_path,imagefilename, low_resl_folder)
                print(f'CREATING LOW RES IMAGES for DISPLAY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


                gray_ad_image = convert_to_grayscale(cropped_path)

                ocr_text_ad = image_ocr(gray_ad_image, f'eng+{language}').strip()

                # ocr_text_ad = do_ocr(cropped_path, f'eng+{language}')
                if len(ocr_text_ad) == 0:
                    ocr_text_ad = ''

                
                folder_name_from_image = imagefilename.split('.')[0]
                print(folder_name_from_image)
                output_folder_logo_segmentation = os.path.join(output_path, folder_name_from_image)
                run_logo_detection_yolo(logo_detection_model, cropped_path, output_folder_logo_segmentation)

                # Path to YOLO-predicted logos
                predict_folder = os.path.join(output_folder_logo_segmentation, "predict", 'crops', 'Logo')

                
                concatenated_logo_paths = None
                # advertising_company = None
                concatenated_company_match_result = None
                concatenated_company_names_unique_sorted = None

                logos_ocr = None

                

                if os.path.exists(predict_folder):
                    filenames = os.listdir(predict_folder)
                    concatenated_logo_paths = "/".join(filenames)
                    matched_company_list = []


                    # if newspaper in ('PS', 'DB', 'HB'):

                    client_data = []
                    logo_ocr_list = []

                    for logo_file in os.listdir(predict_folder):
                        if logo_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            logo_path = os.path.join(predict_folder, logo_file)
                            
                            result = match_logo(logo_path)


                            # convert all predicted logos in grayscale
                            # then do ocr and save in the database

                            gray_logo_image = convert_to_grayscale(logo_path)

                            logo_ocr_result = image_ocr(gray_logo_image, f'eng+{language}').strip()

                            logo_ocr_list.append(logo_ocr_result)



                            if result:
                                company_name = result['company_name']
                                similarity= result['similarity']

                                temp_result = [company_name, similarity]

                                print('temp_result', temp_result)

                                client_data.append(temp_result)
                                
                                matched_company_list.append(f'{company_name} ({similarity*100})')

                            else:
                                copy_images(logo_path, os.path.join(NOT_MATCHED_LOGO_PATH, logo_file))

                        # print(f"Generating CLIP embedding for logo: {logo_path}")
                        # logo_embedding = get_clip_embedding(logo_path)
                        # print("Logo embedding shape:", logo_embedding.shape)
                        # torch.save(logo_embedding, os.path.join(predict_folder, f"{os.path.splitext(logo_file)[0]}_embedding.pt"))
                    if len(matched_company_list) > 0:
                        concatenated_company_match_result = '/'.join(matched_company_list)

                    if len(client_data) > 0:
                        client_data.sort(key= lambda x: x[1], reverse=True)



                        company_names_only_sorted = [item[0] for item in client_data]

                        company_names_unique_sorted = list(dict.fromkeys(company_names_only_sorted))
                        concatenated_company_names_unique_sorted = '/'.join(company_names_unique_sorted)

                        brand_name = None

                    
                    logos_ocr = ('/'.join(logo_ocr_list)).strip()

                    if logos_ocr and logos_ocr != '':
                        pass
                    else:
                        logos_ocr = None


                    copy_images(cropped_path, os.path.join(LOGO_FOUND_ON_THESE_DISPLAY_PATH, imagefilename))

                    print('\n\n\n\nLOGO Company Name ', concatenated_company_match_result, '\n\n\n')
                else:
                    copy_images(cropped_path, os.path.join(ADS_NOT_MAPPED_TO_COMPANY_PATH, imagefilename))
                    print("⚠️ No 'predict' folder found for logo detection.")

                image_name = os.path.basename(cropped_path)       

                height_Display_Ad_cm, width_Display_Ad_cm, area_Display_Ad_cm_sq = dimension_acc_newspaper(image_path, cropped_path, article_in_pixel)

                # ____________________________________________column count_______________________
                col_count = col_counter(width_Display_Ad_cm)

                # display_ad_dim = str(round(height_Display_Ad_cm)) + "~" + str(round(width_Display_Ad_cm)) + "~" + str(round(area_Display_Ad_cm_sq))

                #print("Display Ad Dimension: ",display_ad_dim)
                
                total_Display_Ad_area_cm_sq += area_Display_Ad_cm_sq



                # GETTING BRAND , PARENT_COMPANY, ADVT. CATEGORIES

                data_dict = identify_unknown_brand_using_VLM(cropped_path)

                print('______________datadict_______________________________________________  ',data_dict )

                # Parse the JSON string into a Python dictionary
                # data_dict = json.loads(data_json)

                # Access the data like a standard Python dictionary
                # print(f"Name: {data['name']}")
                # print(f"Age: {data['age']}")
                # print(f"City: {data['city']}")

                brand_name_vlm = None
                parent_company_vlm = None
                ad_category_vlm = None

                try:

                    if data_dict:
                        brand_name_vlm = data_dict['brand']
                        if brand_name_vlm ==  'Unknown':
                            brand_name_vlm = None
                        else:
                            print(f"THIS IS THE BRAND NAME {brand_name_vlm}")

                        parent_company_vlm = data_dict['parent_company']
                        if parent_company_vlm ==  'Unknown':
                            parent_company_vlm = None
                        else:
                            print(f"THIS IS THE COMPANY NAME {parent_company_vlm}")

                        ad_category_vlm = data_dict['category']
                        if ad_category_vlm ==  'Unknown':
                            ad_category_vlm = None

                except Exception as e:
                    logger.exception(f"An error occured during Ollama response, likely a JSON parsing error {newspaper_img}")
                    print(f'\n\n\n\n Excepiton : {e} \n\n\n')



                
                insert_into_advertisements(image_name, folder_name, date, newspaper, edition, subedition, page_no, language, adtype, ocr_text_ad, low_res_path, height_Display_Ad_cm, width_Display_Ad_cm, area_Display_Ad_cm_sq, col_count, concatenated_logo_paths, concatenated_company_match_result, logos_ocr, brand_name_vlm, parent_company_vlm, ad_category_vlm)

                # if brand_name_vlm:
                #     insert_into_imdeia_advt_dash( date, newspaper, edition, subedition, page_no, adtype, height_Display_Ad_cm, width_Display_Ad_cm, area_Display_Ad_cm_sq, parent_company_vlm, brand_name_vlm, ad_category_vlm)
                

        else:
            print(os.path.join(folder_to_check, 'Display'))
            print("not found")
        

        classified_ad_num = 0
        total_classified_Ad_area_cm_sq = 0
        if os.path.exists(os.path.join(folder_to_check, 'Classified')):
            #go to predict 
            print("Foldert Classified exisits")
            print(folder_exists)
            print("Classified Advertisement FOUND")
            predict_path=f'{unq_folder_name}\crops'
            predict_path=os.path.join(predict_path,'Classified')
            folder_name = predict_path.replace('\\', '-')

            output_path= os.path.join(output_folder, predict_path)

            low_resl_folder = os.path.join(output_path, "low_resl_imgs")
            create_folder(low_resl_folder)

            adtype = 'Classified'
            # Extract cropped images and perform OCR
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

            # List all files in the directory and filter by image extensions
            image_files = [f for f in os.listdir(output_path) if f.lower().endswith(image_extensions)]
            for imagefilename in image_files:
                low_res_path = "low_resl_imgs-" + imagefilename
                classified_ad_num += 1
                cropped_path= os.path.join(output_path, imagefilename)

                create_low_res(output_path,imagefilename, low_resl_folder)
                print(f'CREATING LOW RES IMAGES for CLASSIFIED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

                ocr_text_ad = do_ocr(cropped_path, f'eng+{language}')
                if len(ocr_text_ad) == 0:
                    ocr_text_ad = ''

                image_name = os.path.basename(cropped_path)
                print(f"PATH OF INDIVIDUAL CROPPED Classified Advertisement: \n {cropped_path} \n")     

                height_classified_Ad_cm, width_classified_Ad_cm, area_classified_Ad_cm_sq = dimension_acc_newspaper(image_path, cropped_path, article_in_pixel )

                
                # ____________________________________________column count_______________________
                col_count = col_counter(width_classified_Ad_cm)

                # classified_ad_dim = str(round(height_classified_Ad_cm)) + "~" + str(round(width_classified_Ad_cm)) + "~" + str(round(area_classified_Ad_cm_sq))

                #print("Display Ad Dimension: ",classified_ad_dim)
                
                total_classified_Ad_area_cm_sq += area_classified_Ad_cm_sq

                concatenated_logo_paths = None
                advertising_company = None

                insert_into_advertisements(image_name, folder_name, date, newspaper, edition, subedition, page_no, language, adtype, ocr_text_ad, low_res_path, height_classified_Ad_cm, width_classified_Ad_cm, area_classified_Ad_cm_sq, col_count, concatenated_logo_paths, advertising_company, logo_ocr=None, brand_name_vlm = None, parent_company_vlm = None, ad_category_vlm= None)

        else:
            print(os.path.join(folder_to_check, 'Classified'))
            print("not found")

        totalad = display_ad_num + classified_ad_num

        # __________________________________________________-Extras-----------------------------------------------
        Extras_num = 0
        total_Extras_area_cm_sq = 0
        if os.path.exists(os.path.join(folder_to_check, 'Extras')):
            #go to predict 
            print("Foldert Extras exisits")
            print(folder_exists)
            print("Extras FOUND")
            predict_path=f'{unq_folder_name}\crops'
            predict_path=os.path.join(predict_path,'Extras')
            folder_name = predict_path.replace('\\', '-')

            output_path= os.path.join(output_folder, predict_path)

            low_resl_folder = os.path.join(output_path, "low_resl_imgs")
            create_folder(low_resl_folder)

            # Extract cropped images and perform OCR
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

            # List all files in the directory and filter by image extensions
            image_files = [f for f in os.listdir(output_path) if f.lower().endswith(image_extensions)]
            for imagefilename in image_files:
                low_res_path = "low_resl_imgs-" + imagefilename
                Extras_num += 1
                cropped_path= os.path.join(output_path, imagefilename)

                create_low_res(output_path,imagefilename, low_resl_folder)
                print(f'CREATING LOW RES IMAGES for Extras>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

                ocr_text = do_ocr(cropped_path, f'eng+{language}')
                if len(ocr_text) == 0:
                    ocr_text = ''

                image_name = os.path.basename(cropped_path)
                print(f"PATH OF INDIVIDUAL CROPPED extras : \n {cropped_path} \n")     
                
                height_Ex_cm, width_Ex_cm , area_extras_cm_sq  = dimension_acc_newspaper(image_path, cropped_path, article_in_pixel )

                
                # ____________________________________________column count_______________________
                column_count = col_counter(width_Ex_cm)

                # classified_ad_dim = str(round(height_classified_Ad_cm)) + "~" + str(round(width_classified_Ad_cm)) + "~" + str(round(area_classified_Ad_cm_sq))

                #print("Display Ad Dimension: ",classified_ad_dim)
                total_Extras_area_cm_sq += area_extras_cm_sq 
                # createdAt = datetime.now()
                # updatedAt = createdAt
                insert_into_extras( folder_name, image_name, date, newspaper, edition, subedition, page_no, language, ocr_text, height_Ex_cm,  width_Ex_cm,  area_extras_cm_sq,  column_count )
            
        else:
            print(os.path.join(folder_to_check, 'Extras'))
            print("not found")
        # __________________________________________________-Extras-----------------------------------------------
        
        if article_num == 0:
            article_num = None

        if total_article_area_cm_sq == 0:
            total_article_area_cm_sq = None
        
        if display_ad_num == 0:
            display_ad_num = None
            
        if total_Display_Ad_area_cm_sq == 0:
            total_Display_Ad_area_cm_sq = None
        
        if classified_ad_num == 0:
            classified_ad_num = None
        
        if total_classified_Ad_area_cm_sq == 0:
            total_classified_Ad_area_cm_sq =None

        if image_num_per_page == 0:
            image_num_per_page = None
        
        if image_area_per_page == 0:
            image_area_per_page= None
                
        
        insert_into_table_pagesummaries(newspaper, edition, subedition, date, page_no,  article_num, display_ad_num, classified_ad_num, totalad, total_article_area_cm_sq, total_Display_Ad_area_cm_sq, total_classified_Ad_area_cm_sq, image_num_per_page, image_area_per_page)


        move_imgs_to_done(newspaper_imgs_path, 'Done', newspaper_img)
        logger.info(f"{newspaper_img} is moved to Done\n")
    
    except Exception as e:
        print("--------------------------------------------------------------------------------------------------")
        print(newspaper_img)
        print(e)
        print("--------------------------------------------------------------------------------------------------")
              
        logger.exception(f"A critical error occurred during math operations {newspaper_img}")

if __name__ == "__main__":

    imgs_path = r"C:\Article-Detection-App\NEWSPAPERS_DL\PAGE_IMGS"
    arg_list = []
    for newspaper_img in os.listdir(imgs_path):
        if newspaper_img.lower().endswith((".jpg", ".png", ".jpeg")):
            arg_list.append(newspaper_img)

    # ✅ Prioritize filenames containing 'csnagar' (case-insensitive)
    arg_list.sort(key=lambda x: "csnagar" not in x.lower())


    with Pool(processes=1) as pool:
        pool.map(main, arg_list)

    print("DONE............")
    
