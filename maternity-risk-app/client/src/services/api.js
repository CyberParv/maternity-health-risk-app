import axios from 'axios';

// API configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// API endpoints
export const predictRisk = async (patientData) => {
  try {
    console.log('Sending prediction request to backend');
    const response = await api.post('/predict', patientData);
    return response.data;
  } catch (error) {
    console.error('Error making prediction:', error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Error getting model info:', error);
    throw error;
  }
};

export default api; 