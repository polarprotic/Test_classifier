"""
Standalone tester for VLM brand/parent-company identification.

Usage:
    python test_brand_identification.py

Edit INPUT_DIR / OUTPUT_DIR / OLLAMA_URL below, or override via:
    python test_brand_identification.py --input ./test_input --output ./test_output --ocr

Reads every image in INPUT_DIR, calls the local Ollama VLM, and copies each
image into OUTPUT_DIR renamed as BRAND__COMPANY.<ext>. Also writes
results.json in the output folder with the raw model output, so you can
audit accuracy without opening every filename.
"""

import os
import re
import json
import time
import shutil
import base64
import argparse
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:8b"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# ---- Edit these and just run the script directly, no command-line args needed ----
INPUT_DIR = "test_input"
OUTPUT_DIR = "test_output"
USE_OCR = True
OCR_LANG = "eng"
# ------------------------------------------------------------------------------


def sanitize(name, fallback="Unknown"):
    if not name or not str(name).strip():
        return fallback
    name = str(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:60] if name else fallback


def get_ocr_hint(image_path, lang="eng"):
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(image_path), lang=lang).strip()
    except Exception as e:
        print(f"  [OCR skipped] {e}")
        return ""


def extract_json(raw):
    """
    Try to pull a usable {brand, parent_company, category} dict out of the
    model's raw text, even if it wrapped the JSON in markdown, added
    explanation before/after it, or produced slightly malformed JSON.
    """
    if not raw:
        return None

    text = raw.strip()

    # 1. Try straight parse first (works when format=json behaves)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences if present
    fenced = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # 3. Find the first {...} block anywhere in the text (handles models that
    #    add reasoning/explanation before or after the JSON object)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # try fixing common issues: trailing commas, single quotes
            fixed = re.sub(r",\s*}", "}", candidate)
            fixed = fixed.replace("'", '"')
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # 4. Last resort: pull brand/parent_company/category via regex even
    #    without valid JSON structure, so partial info isn't lost
    result = {}
    for field in ("brand", "parent_company", "category"):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
        if m:
            result[field] = m.group(1)
    return result if result else None


def call_ollama(payload, attempt=1, max_attempts=2):
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        print(f"  [HTTP {resp.status_code}] (attempt {attempt})")
        if resp.status_code >= 400:
            print(f"  [RESPONSE BODY] {resp.text[:500]}")
        resp.raise_for_status()
        return resp.json().get("response", "{}"), None
    except requests.exceptions.ConnectionError as e:
        msg = f"CONNECTION ERROR: {e}"
    except requests.exceptions.Timeout as e:
        msg = f"TIMEOUT: {e}"
    except requests.exceptions.HTTPError as e:
        msg = f"HTTP ERROR: {e}"
    except Exception as e:
        msg = f"UNEXPECTED ERROR: {type(e).__name__}: {e}"

    if attempt < max_attempts:
        print(f"  [RETRYING] {msg}")
        time.sleep(2)
        return call_ollama(payload, attempt + 1, max_attempts)

    print(f"  [FAILED after {attempt} attempts] {msg}")
    return None, msg


def identify_brand(image_path, ocr_hint=""):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""Look at this newspaper ad image and identify ONE brand and ONE parent company.

Rules:
- brand = the single most prominent product/service name (largest text, main headline, primary logo).
- parent_company = the company that owns that brand, if shown (often smaller text/logo). If not shown, use "Unknown".
- If the image shows MULTIPLE sponsors/logos (event banners, co-presented-by, powered-by ads), pick ONLY the single most prominent one (largest, top, or center) as brand. Do NOT list multiple names. Do NOT explain your choice.
- If nothing is identifiable, use "Unknown" for both.
- category = one or two words describing what's being advertised (e.g. "education", "baby products", "real estate").

OCR hint (may be inaccurate, image is the source of truth): "{ocr_hint}"

Respond with EXACTLY this JSON and nothing else — no reasoning, no markdown, no extra text before or after:
{{"brand": "...", "parent_company": "...", "category": "..."}}"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        # NOTE: "format": "json" deliberately omitted — forcing JSON-grammar
        # output degraded answers for this model (truncated/empty fields).
        # The prompt instructs JSON output, and extract_json() below handles
        # pulling it out even if wrapped in extra text.
        "options": {
            "temperature": 0,
            "num_ctx": 8192,  # default 4096 was too small for large/dense images, causing HTTP 400
        },
    }

    raw = None
    try:
        raw, err = call_ollama(payload)
        if raw is None:
            return {"brand": "ERROR", "parent_company": "ERROR", "category": "ERROR"}

        print(f"  [RAW MODEL OUTPUT] {raw[:300]}")

        data = extract_json(raw)
        if data is None:
            print(f"  [JSON PARSE ERROR] Could not extract anything usable from model output")
            return {"brand": "ERROR", "parent_company": "ERROR", "category": "ERROR"}

        if data.get("brand") and data.get("brand") == data.get("parent_company"):
            data["parent_company"] = "Unknown"
        return data
    except Exception as e:
        print(f"  [UNEXPECTED ERROR] {type(e).__name__}: {e}")
        return {"brand": "ERROR", "parent_company": "ERROR", "category": "ERROR"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DIR, help="Folder of input images")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder to write renamed copies + results")
    parser.add_argument("--ocr", action="store_true", default=USE_OCR, help="Run pytesseract OCR and feed as hint")
    parser.add_argument("--lang", default=OCR_LANG, help="OCR language (default eng)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = []

    images = [f for f in os.listdir(args.input) if f.lower().endswith(VALID_EXTENSIONS)]
    print(f"Found {len(images)} images in {args.input}")
    print(f"Using Ollama at: {OLLAMA_URL}\n")

    for idx, filename in enumerate(images, 1):
        src_path = os.path.join(args.input, filename)
        ext = os.path.splitext(filename)[1]
        print(f"[{idx}/{len(images)}] {filename}")

        ocr_hint = get_ocr_hint(src_path, args.lang) if args.ocr else ""
        data = identify_brand(src_path, ocr_hint)

        brand = sanitize(data.get("brand"))
        company = sanitize(data.get("parent_company"))

        new_name = f"{brand}__{company}{ext}"
        dest_path = os.path.join(args.output, new_name)

        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(args.output, f"{brand}__{company}_{counter}{ext}")
            counter += 1

        shutil.copy2(src_path, dest_path)
        print(f"  -> {os.path.basename(dest_path)}\n")

        results.append({
            "original_filename": filename,
            "new_filename": os.path.basename(dest_path),
            "brand": data.get("brand"),
            "parent_company": data.get("parent_company"),
            "category": data.get("category"),
            "ocr_hint_used": ocr_hint[:200] if ocr_hint else "",
        })

    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(results)} images processed.")
    print(f"Renamed copies + results.json saved to: {args.output}")


if __name__ == "__main__":
    main()