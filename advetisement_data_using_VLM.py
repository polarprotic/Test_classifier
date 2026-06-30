# import ollama 

# # Inference using a local image file path
# response = ollama.chat(
#     model='qwen3-vl:8b',
#     messages=[{
#         'role': 'user',
#         'content': '''Analyze this image of a advertisement from a newspaper.
#                 1. Identify the brand name.
#                 2. Identify the parent company (if applicable).
#                 3. Describe the product or service being advertised.''',
#         'images': ['C:\Article-Detection-App\display logo\AU_Noida_Main_190325_hin_22_.jpg'] # Path to your image
#     }]
# )

# print(response['message']['content'])




# import ollama
# import json

# def identify_unknown_brand(image_path):
#     """
#     Triggers a VLM to identify a logo that CLIP/Vector Search couldn't find.
#     """
#     prompt = """
#     Analyze this image of a logo/advertisement from a newspaper. 
#     1. Identify the brand name.
#     2. Identify the parent company (if applicable).
#     3. Describe the product or service being advertised.
    
#     Return the result in JSON format:
#     {"brand": "...", "parent_company": "...", "category": "..."}
#     """
    
#     # Use 'qwen2.5-vl' or 'llama3.2-vision' for high accuracy
#     response = ollama.chat(
#         model='qwen3-vl:8b',
#         messages=[{
#             'role': 'user',
#             'content': prompt,
#             'images': [image_path]
#         }],
#         host='http://192.168.3.138:11434'
#     )

    
    
#     # Extract the text and parse JSON (cleaning the string if needed)
#     try:
#         data = json.loads(response['message']['content'])
#         return data
#     except:
#         return response['message']['content']

# data = identify_unknown_brand(r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\AU_Noida_Main_190325_hin_6_2.jpg')


# print(data)











# import requests

# url = "http://192.168.3.138:11434/api/chat"


# prompt = """
#     Analyze this image of a logo/advertisement from a newspaper. 
#     1. Identify the brand name.
#     2. Identify the parent company (if applicable).
#     3. Describe the product or service being advertised.
    
#     Return the result in JSON format:
#     {"brand": "...", "parent_company": "...", "category": "..."}
#     """

# image_path = r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\AU_Noida_Main_190325_hin_6_2.jpg'

# payload = {
#     "model": "qwen3-vl:8b",
#     "messages": [
#         {
#             "role": "user",
#             "content": prompt,
#             'images': [image_path]
#         }
#     ]
# }

# response = requests.post(url, json=payload)
# print(response.json())


# import requests

# url = "http://192.168.3.138:11434/api/chat"

# prompt = """
#     Analyze this image of a logo/advertisement from a newspaper. 
#     1. Identify the brand name.
#     2. Identify the parent company (if applicable).
#     3. Describe the product or service being advertised.
    
#     Return the result in JSON format:
#     {"brand": "...", "parent_company": "...", "category": "..."}
# """

# image_path = r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\AU_Noida_Main_190325_hin_6_2.jpg'

# # Open the image file in binary mode for uploading
# with open(image_path, 'rb') as img_file:
#     files = {
#         'image': img_file  # Upload the image file
#     }

#     # Prepare the message content
#     data = {
#         "model": "qwen3_vl:8b",  # Ensure this is a valid model identifier in the API
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     }

#     # Send the POST request with both the data and the image file
#     response = requests.post(url, data=data, files=files)

# # Check for response content and handle possible errors
# try:
#     response_data = response.json()  # Parse JSON response
#     print(response_data)
# except ValueError:
#     print("Error parsing response:", response.text)

# #  ----------------------------------------------------------------------------------------------

# # Replace <host-ip-address> with the actual IP
# client = ollama.Client(host='http://4c-desktop:11434')

# # image_path = r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\DIVY_CSNagar_Sub_180326_mar_1_.jpg'

# def identify_unknown_brand_using_VLM(image_path):
#     prompt = """
#         Analyze this image of a advertisement from a newspaper. 
#         1. Identify the brand name(if applicable).
#         2. Identify the parent company (if applicable).
#         3. Describe the product or service being advertised.
#         4. If something not found, respond with - 'Unknown'.
        
#         Return the result in JSON format:
#         {"brand": "...", "parent_company": "...", "category": "..."}
#     """

#     response = client.chat(model='qwen3-vl:8b', messages=[
#         {
#             'role': 'user',
#             'content': prompt,
#             'images': [image_path]
#         },
#     ])
#     return (response['message']['content'])




# ----------------------------------------------------


# import requests
# import json

# def identify_unknown_brand_using_VLM(image_path):
#     prompt = """
#         Analyze this image of a advertisement from a newspaper. 
#         1. Identify the brand name(if applicable).
#         2. Identify the parent company (if applicable).
#         3. Describe the product or service being advertised.
#         4. If something not found, respond with - 'Unknown'.
        
#         Return the result in JSON format:
#         {"brand": "...", "parent_company": "...", "category": "..."}
#     """

#     url = 'http://4c-desktop:11434/api/chat'
    
#     payload = {
#         'role': 'user',
#         "model": "qwen3-vl:8b",
#         "content": prompt,
#         "images": [image_path]
        
#     }
    
#     response = requests.post(url, json=payload)
    
#     # # Parse JSON response
#     # return response.json()["response"]



#     print(response.status_code)       # Should be 200
#     print(response.text)              # Raw response
#     print(response.json())            # Parsed JSON


# # print(identify_unknown_brand_using_VLM(r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\DIVY_CSNagar_Sub_180326_mar_1_.jpg'))

# identify_unknown_brand_using_VLM(r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\DIVY_CSNagar_Sub_180326_mar_1_.jpg')



import requests
import json
import base64

def identify_unknown_brand_using_VLM(image_path):
    # 1. Encode the image to Base64
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return "Error: Image file not found."

    prompt = """
        Analyze this image of an advertisement from a newspaper. 
        1. Identify the brand name (if applicable).
        2. Identify the parent company (if applicable).
        3. Describe the product or service being advertised.
        4. If something is not found, respond with - 'Unknown'.
        
        Return the result ONLY in JSON format:
        {"brand": "...", "parent_company": "...", "category": "..."}
    """

    url = 'http://localhost:11434/api/generate' # Changed to /generate for simpler output
    
    payload = {
        "model": "qwen3-vl:8b", # Ensure this matches your local model name
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        
    }
    # "format": "json" # Forces the model to output valid JSON
    
    try:
        response = requests.post(url, json=payload)

        print('وَعَلَيْكُمُ ٱلسَّلَامُ  💣💥', response)
        response.raise_for_status()
        
        # Parse and return the actual content
        result = response.json()
        return json.loads(result.get("response", "{}"))
        
    except Exception as e:
        return f"An error occurred: {e}"

# # Example Execution
# image_loc = r'C:\Article-Detection-App\SCRIPTS\ADS_NOT_MAPPED_TO_COMPANY\DIVY_CSNagar_Sub_180326_mar_1_.jpg'
# data = identify_unknown_brand_using_VLM(image_loc)
# print(json.dumps(data, indent=4))