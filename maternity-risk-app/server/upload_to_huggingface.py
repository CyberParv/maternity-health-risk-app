import os
import pickle
import json
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("HF_USERNAME:", os.getenv("HF_USERNAME"))
print("HF_TOKEN starts with:", os.getenv("HF_TOKEN")[:5])  # Just the start for safety

# Initialize Hugging Face API
api = HfApi()

def create_model_card(model_name, model_dir):
    """Create a model card with documentation."""
    card_content = f"""---
language: en
license: mit
tags:
- maternity
- health
- risk-prediction
---

# Maternity Health Risk Prediction Model

This model predicts the risk level for maternity health based on various medical indicators.

## Model Description

- **Model Type**: {model_name}
- **Task**: Binary Classification (Low/High Risk)
- **Input Features**: 
  - Systolic BP
  - Diastolic BP
  - Blood Sugar
  - BMI
  - Heart Rate
  - Previous Complications
  - Preexisting Diabetes
  - Gestational Diabetes
  - Mental Health

## Usage

```python
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input features
features = np.array([[
    120,  # Systolic BP
    80,   # Diastolic BP
    100,  # Blood Sugar
    25,   # BMI
    75,   # Heart Rate
    0,    # Previous Complications
    0,    # Preexisting Diabetes
    0,    # Gestational Diabetes
    0     # Mental Health
]])

# Make prediction
prediction = model.predict(features)
```
"""
    
    with open(f"{model_dir}/README.md", "w") as f:
        f.write(card_content)

def main():
    # Create repository if it doesn't exist
    repo_name = "maternity-health-risk"
    try:
        create_repo(f"{os.getenv('HF_USERNAME')}/{repo_name}", private=False)

    except Exception as e:
        print(f"Repository might already exist: {e}")

    
    # Convert and upload models
    models_dir = "../models"

    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pkl"):
            model_name = model_file.replace(".pkl", "")
            model_path = os.path.join(models_dir, model_file)
            
            print(f"Processing {model_name}...")
            
            # Create a directory for this model
            model_dir = f"hf_models/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy the model file
            with open(model_path, 'rb') as src, open(f"{model_dir}/model.pkl", 'wb') as dst:
                dst.write(src.read())
            
            # Create model card
            print(f"Creating model card for {model_name}...")
            create_model_card(model_name, model_dir)
            
            # Upload to Hugging Face
            print(f"Uploading {model_name} to Hugging Face...")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=f"{os.getenv('HF_USERNAME')}/{repo_name}",

                repo_type="model"
            )
            
            print(f"Successfully uploaded {model_name}")

if __name__ == "__main__":
    main() 