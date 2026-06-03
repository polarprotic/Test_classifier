import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FOLDER = "test_images"       
MODEL_PATH = "newspaper_classifier_3_best_again.pt" 

CLASS_NAMES = {
    0: "Article",
    1: "Classified",
    2: "Display"
}

# ==========================================
# 2. MODEL ARCHITECTURE & LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the exact EfficientNet architecture
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

# Load your saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Set the model to evaluation mode
model.eval()
model.to(device)

# Standard preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. PREDICTION & TERMINAL PRINTING LOGIC
# ==========================================
def predict_images():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp') 
    processed_count = 0

    print(f"Scanning '{INPUT_FOLDER}' for images...\n")
    
    # Expanded the column width to 70 to fit longer file paths comfortably
    print(f"{'IMAGE PATH':<70} | {'PREDICTION':<15} | {'CONFIDENCE'}")
    print("-" * 105)

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        file_path = os.path.join(INPUT_FOLDER, filename)
        # Get the full absolute path of the image on your computer
        full_path = os.path.abspath(file_path)
        
        try:
            # Load and prepare the image
            image = Image.open(file_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Make the prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                
                # Apply softmax to convert raw outputs to probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get the highest probability and the index of that class
                max_prob, predicted_idx = torch.max(probabilities, 0)
            
            # Convert probability to a whole number percentage
            confidence_pct = int(round(max_prob.item() * 100))
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            
        
            print(f"{full_path:<70} | {predicted_class:<15} | {confidence_pct}%")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

    print(f"\nFinished! Processed {processed_count} images.")

if __name__ == "__main__":
    predict_images()