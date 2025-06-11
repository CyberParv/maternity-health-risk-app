const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const axios = require('axios');

// Initialize express app
const app = express();
const PORT = process.env.PORT || 5000;
// Use environment variable for the Python API URL
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:5001/predict';

// Middleware
// Configure CORS to allow requests from Vercel and other origins
app.use(cors({
  // Allow all origins for development and testing
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  credentials: true,
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Set additional CORS headers manually to ensure they're set
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.header('Access-Control-Allow-Credentials', 'true');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path} - Origin: ${req.headers.origin || 'unknown'}`);
  next();
});

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the React app in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/build')));
}

// Function to call Python model API
async function callModelAPI(data) {
  try {
    const response = await axios.post(PYTHON_API_URL, data);
    return response.data;
  } catch (error) {
    console.error('Error calling Python API:', error);
    // Fall back to rule-based prediction if Python API is unavailable
    return generateFallbackPrediction(data);
  }
}

// Fallback rule-based prediction (used if Python API fails)
function generateFallbackPrediction(data) {
  const {
    systolic_bp,
    diastolic,
    bs,
    bmi,
    heart_rate,
    previous_complications,
    preexisting_diabetes,
    gestational_diabetes,
    mental_health
  } = data;
  
  // Generate rule-based prediction
  const isHighRisk = 
    preexisting_diabetes === 1 || 
    gestational_diabetes === 1 || 
    (mental_health === 1 && previous_complications === 1) ||
    (systolic_bp > 140 && diastolic > 90);
  
  // Calculate probability based on risk factors
  let probability = 0.5; // base
  
  if (preexisting_diabetes === 1) probability += 0.25;
  if (gestational_diabetes === 1) probability += 0.15;
  if (mental_health === 1) probability += 0.1;
  if (previous_complications === 1) probability += 0.1;
  if (systolic_bp > 140) probability += 0.05;
  if (diastolic > 90) probability += 0.05;
  if (bmi > 30) probability += 0.05;
  
  // Cap probability at 0.95
  probability = Math.min(probability, 0.95);
  
  // Feature importance data
  const feature_importance = {
    'Preexisting Diabetes': preexisting_diabetes === 1 ? 0.35 : 0.05,
    'Mental Health': mental_health === 1 ? 0.25 : 0.05,
    'Gestational Diabetes': gestational_diabetes === 1 ? 0.15 : 0.05,
    'Previous Complications': previous_complications === 1 ? 0.10 : 0.05,
    'BMI': 0.05,
    'Heart Rate': 0.03,
    'Blood Sugar': 0.03,
    'Systolic BP': 0.02,
    'Diastolic BP': 0.02,
  };
  
  return {
    prediction: isHighRisk ? 1 : 0,
    probability: isHighRisk ? probability : 1 - probability,
    feature_importance,
    model_used: 'Rule-based Fallback'
  };
}

// API endpoint for risk prediction
app.post('/api/predict', async (req, res) => {
  try {
    console.log('Received prediction request:', req.body);
    
    // Call the model API
    const result = await callModelAPI(req.body);
    
    // Return prediction result
    res.json(result);
  } catch (error) {
    console.error('Error in prediction:', error);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Maternity risk API server is running' });
});

// Catch-all route in production to serve React app
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
  });
}

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 