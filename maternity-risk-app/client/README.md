# Maternity Risk Client

This is the frontend for the Maternity Health Risk Application.

## Connecting Vercel Frontend to Local Backend

Follow these steps to deploy the frontend to Vercel while keeping the backend running locally:

### 1. Set Up Local Backend

1. Start your local Node.js server:
   ```
   cd maternity-risk-app/server
   npm install
   node server.js
   ```

2. Start your local Python API:
   ```
   cd maternity-risk-app/server
   pip install -r requirements.txt
   python predict_api.py
   ```

### 2. Deploy Frontend to Vercel

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Navigate to the client directory:
   ```
   cd maternity-risk-app/client
   ```

3. Deploy to Vercel:
   ```
   npm run deploy-vercel
   ```
   
   Or manually:
   ```
   vercel --prod
   ```

### 3. Expose Local Backend to Internet (Required for Testing with Vercel)

Since Vercel-hosted frontend needs to access your local backend, you need to expose your local servers. Options include:

#### Option 1: Use Ngrok

1. Install ngrok:
   ```
   npm install -g ngrok
   ```

2. Expose your Node.js server:
   ```
   ngrok http 5000
   ```

3. Expose your Python API:
   ```
   ngrok http 5001
   ```

4. Update environment variables in Vercel:
   - Go to your Vercel project dashboard
   - Navigate to Settings > Environment Variables
   - Add:
     - `REACT_APP_API_URL=https://<your-ngrok-url-for-node>/api`
     - `REACT_APP_PYTHON_API_URL=https://<your-ngrok-url-for-python>`

#### Option 2: Use LocalTunnel

1. Install localtunnel:
   ```
   npm install -g localtunnel
   ```

2. Expose your Node.js server:
   ```
   lt --port 5000
   ```

3. Expose your Python API:
   ```
   lt --port 5001
   ```

4. Update environment variables in Vercel as described above

### 4. Access Your Application

- Frontend: https://your-vercel-app-url.vercel.app
- Backend: Your local servers (via ngrok or localtunnel)

## Important Notes

- This setup is recommended for development/testing only
- For production, consider deploying all components to cloud services
- Remember CORS settings in your backend may need adjustment
- The tunneling services may have connection limits or timeouts 