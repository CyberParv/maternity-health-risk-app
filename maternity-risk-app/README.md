# Maternity Health Risk Application

This application predicts maternity health risks using machine learning models. It consists of:
- React frontend
- Node.js API server
- Python ML prediction service

## Deploying to Vercel and Other Cloud Platforms

### 1. Frontend Deployment (Vercel)

The React frontend can be deployed to Vercel:

1. Install Vercel CLI: `npm install -g vercel`
2. Navigate to the client directory: `cd maternity-risk-app/client`
3. Deploy to Vercel: `vercel`
4. Follow the prompts to complete deployment

Alternatively, connect your GitHub repository to Vercel and configure:
- Framework preset: Create React App
- Build command: `npm run build`
- Output directory: `build`
- Install command: `npm install`

### 2. Node.js Server Deployment (Heroku/Railway/Render)

1. Create a new app on your chosen platform
2. Set environment variables:
   - `PYTHON_API_URL`: URL to your Python API
   - `FRONTEND_URL`: URL of your Vercel-deployed frontend
   - `NODE_ENV`: Set to `production`
3. Deploy the `maternity-risk-app/server` directory
4. Make sure to exclude the `models` directory from deployment (add to .gitignore)

### 3. Python ML API Deployment (Heroku/Railway/PythonAnywhere)

1. Create a new app on your chosen platform
2. Set up your Python environment (Python 3.8+ recommended)
3. Make sure the `models` directory with all pickle files is included
4. Set environment variables as needed
5. Deploy the Python API
6. Note: The ML models might require substantial memory, choose an appropriate tier

### Environment Variables

#### Frontend (Vercel)
- `REACT_APP_API_URL`: URL to your Node.js API server
- `REACT_APP_PYTHON_API_URL`: URL to your Python API (if needed for direct access)

#### Node.js Server
- `PORT`: Port to run the server (platform usually sets this automatically)
- `PYTHON_API_URL`: URL to your deployed Python API
- `FRONTEND_URL`: URL of your frontend for CORS configuration

#### Python API
- `PORT`: Port to run the API (platform usually sets this automatically)

## Local Development

1. Install frontend dependencies: `cd client && npm install`
2. Install backend dependencies: `cd server && npm install`
3. Install Python dependencies: `pip install -r server/requirements.txt`
4. Start frontend: `cd client && npm start`
5. Start Node.js server: `cd server && node server.js`
6. Start Python API: `cd server && python predict_api.py`

## Application Architecture

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  React Client  │────>│  Node.js API   │────>│   Python ML    │
│    (Vercel)    │<────│    Server      │<────│     API        │
└────────────────┘     └────────────────┘     └────────────────┘
```

The application follows a three-tier architecture:
1. **Frontend (React)**: User interface deployed on Vercel
2. **API Server (Node.js)**: Handles API requests and serves as proxy to ML API
3. **ML Service (Python Flask)**: Runs the machine learning models

## Architecture

The application consists of three main components:

1. **Machine Learning Model**: A Python-based ensemble model that combines multiple algorithms for accurate prediction
2. **Backend API**: An Express.js server that interfaces with the machine learning model
3. **Frontend**: A React-based user interface for data input and visualization

## Setup and Installation

### Prerequisites

- Node.js (v14 or higher)
- Python (v3.7 or higher)
- npm or yarn

### Install Dependencies

1. **Train the ML model**:
   ```
   python maternity_risk_analysis.py
   ```
   This will create the necessary model files in the `models` directory.

2. **Install server dependencies**:
   ```
   cd maternity-risk-app/server
   npm install
   pip install -r requirements.txt
   ```

3. **Install client dependencies**:
   ```
   cd maternity-risk-app/client
   npm install
   ```

## Running the Application

1. **Start the Python API** (in one terminal):
   ```
   cd maternity-risk-app/server
   python predict_api.py
   ```
   This starts the Flask API on port 5001.

2. **Start the Express server** (in another terminal):
   ```
   cd maternity-risk-app/server
   npm run dev
   ```
   This starts the Express server on port 5000.

3. **Start the React client** (in a third terminal):
   ```
   cd maternity-risk-app/client
   npm start
   ```
   This starts the React application on port 3000.

4. **Access the application** at: http://localhost:3000

### Running with a single command

Alternatively, you can run both the server and Python API with a single command:

```
cd maternity-risk-app/server
npm run dev-all
```

Then start the client separately:

```
cd maternity-risk-app/client
npm start
```

## Model Information

The maternity risk prediction uses an ensemble of several machine learning models:

1. **Random Forest**: Best individual performance with 98% accuracy
2. **Stack Ensemble**: Meta-model combining predictions from all base models
3. **Weighted Voting Ensemble**: Weighted voting based on validation performance

The ensemble approach provides robustness and higher accuracy than individual models.

## Key Features

- Interactive risk prediction form
- Detailed results with feature importance visualization
- Dashboard for analytics and trend monitoring
- Information about the model and its performance

## Fallback Mechanisms

The system has multiple fallback mechanisms:

1. If the Python API fails, the Express server uses a rule-based prediction
2. If the ensemble model fails, it falls back to weighted voting
3. If all else fails, it uses the Random Forest model 