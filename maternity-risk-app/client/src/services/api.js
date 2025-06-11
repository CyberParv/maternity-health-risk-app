import axios from 'axios';

// API configuration
const HF_API_URL = process.env.REACT_APP_HF_API_URL || 'https://api-inference.huggingface.co/models/CyberParv/maternity-health-risk';
const HF_TOKEN = process.env.REACT_APP_HF_TOKEN;

// Create axios instance with default config
const api = axios.create({
  baseURL: HF_API_URL,
  headers: {
    'Authorization': `Bearer ${HF_TOKEN}`,
    'Content-Type': 'application/json'
  }
});

// API endpoints
export const predictRisk = async (patientData) => {
  try {
    console.log('Sending prediction request to Hugging Face API');
    const response = await api.post('', {
      inputs: {
        systolic_bp: patientData.systolic_bp,
        diastolic: patientData.diastolic,
        bs: patientData.bs,
        bmi: patientData.bmi,
        heart_rate: patientData.heart_rate,
        previous_complications: patientData.previous_complications,
        preexisting_diabetes: patientData.preexisting_diabetes,
        gestational_diabetes: patientData.gestational_diabetes,
        mental_health: patientData.mental_health
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error making prediction:', error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('');
    return response.data;
  } catch (error) {
    console.error('Error getting model info:', error);
    throw error;
  }
};

export default api; 