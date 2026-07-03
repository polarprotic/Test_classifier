
import requests
import json
import base64



CATEGORIES = [
    "Automobiles",
    "Real Estate",
    "Retail & Consumer Goods",
    "Food & Beverages",
    "Healthcare",
    "Education",
    "Jobs & Recruitment",
    "Finance & Banking",
    "Telecom & Technology",
    "Travel & Hospitality",
    "Events & Entertainment",
    "Government & Public Notices",
    "Matrimonial",
    "Classifieds",
    "Industrial & B2B",
    "Lifestyle & Personal Care",
    "Energy & Utilities"
]


def identify_unknown_brand_using_VLM(image_path):
    # 1. Encode the image to Base64
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return "Error: Image file not found."

    prompt = f"""Look at this newspaper ad image and identify ONE brand, ONE parent company, and ONE category.

Rules:
- brand = the single most prominent product/service name (largest text, main headline, primary logo).
- parent_company = the company that owns that brand, if shown (often smaller text/logo). If not shown, use "Unknown".
- If the image shows MULTIPLE sponsors/logos (event banners, co-presented-by, powered-by ads), pick ONLY the single most prominent one (largest, top, or center) as brand. Do NOT list multiple names. Do NOT explain your choice.
- If nothing is identifiable, use "Unknown" for brand and parent_company.
- category = classify the ad into EXACTLY ONE category from this list (do NOT invent new categories, do NOT combine multiple): {CATEGORIES}


Respond with EXACTLY this JSON and nothing else — no reasoning, no markdown, no extra text before or after:
{{"brand": "...", "parent_company": "...", "category": "..."}}"""

    url = 'http://192.168.3.138:11434/api/generate' # Changed to /generate for simpler output
    
    payload = {
        "model": "qwen3-vl:8b", # Ensure this matches your local model name
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0,   
            "num_ctx": 8192,
        },
        
    }
    # "format": "json" # Forces the model to output valid JSON
    
    try:
        response = requests.post(url, json=payload)

        print('وَعَلَيْكُمُ ٱلسَّلَامُ  💣💥', response)
        response.raise_for_status()
        
        # Parse and return the actual content
        result = response.json()
        raw_output = result.get("response", "").strip()
        try:
            return json.loads(result.get("response", "{}"))
        except:
            # fallback cleanup
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        
    except Exception as e:
        return f"An error occurred: {e}"