from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/CyberParv/maternity-health-risk"
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Prepare input for Hugging Face API
        inputs = {
            "systolic_bp": data.get("systolic_bp"),
            "diastolic": data.get("diastolic"),
            "bs": data.get("bs"),
            "bmi": data.get("bmi"),
            "heart_rate": data.get("heart_rate"),
            "previous_complications": data.get("previous_complications"),
            "preexisting_diabetes": data.get("preexisting_diabetes"),
            "gestational_diabetes": data.get("gestational_diabetes"),
            "mental_health": data.get("mental_health")
        }
        
        # Call Hugging Face API
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": inputs})
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Prediction failed"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 