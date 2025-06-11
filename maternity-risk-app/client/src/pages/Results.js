import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  Divider,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PriorityHighIcon from '@mui/icons-material/PriorityHigh';
import FileDownloadIcon from '@mui/icons-material/FileDownload';

function Results() {
  const [result, setResult] = useState(null);
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // In a real application, we would fetch this data from an API
    // Here we retrieve it from sessionStorage (set in PredictionForm)
    const storedResult = sessionStorage.getItem('predictionResult');
    const storedPatientData = sessionStorage.getItem('patientData');

    if (storedResult && storedPatientData) {
      try {
        setResult(JSON.parse(storedResult));
        setPatientData(JSON.parse(storedPatientData));
      } catch (err) {
        setError('Unable to parse prediction result. Please try again.');
        console.error('Error parsing result:', err);
      }
    } else {
      setError('No prediction result found. Please complete the prediction form first.');
    }

    setLoading(false);
  }, []);

  const handleNewPrediction = () => {
    navigate('/predict');
  };

  const handleDownloadPDF = () => {
    // This would generate and download a PDF report in a real application
    alert('PDF report would be downloaded in a real application.');
  };

  // Prepare feature importance data for bar chart
  const prepareFeatureImportanceData = (featureImportance) => {
    return Object.entries(featureImportance)
      .map(([feature, value]) => ({
        feature,
        value: parseFloat(value.toFixed(2)),
      }))
      .sort((a, b) => b.value - a.value);
  };

  if (loading) {
    return (
      <Container maxWidth="md">
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md">
        <Paper elevation={3} sx={{ p: 4, mt: 3 }}>
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
          <Button
            variant="contained"
            onClick={handleNewPrediction}
            startIcon={<ArrowBackIcon />}
          >
            Go to Prediction Form
          </Button>
        </Paper>
      </Container>
    );
  }

  if (!result) {
    return (
      <Container maxWidth="md">
        <Paper elevation={3} sx={{ p: 4, mt: 3 }}>
          <Alert severity="warning">
            No prediction data available. Please complete the prediction form.
          </Alert>
          <Button
            variant="contained"
            onClick={handleNewPrediction}
            sx={{ mt: 2 }}
            startIcon={<ArrowBackIcon />}
          >
            Go to Prediction Form
          </Button>
        </Paper>
      </Container>
    );
  }

  const riskLevel = result.prediction === 1 ? 'High' : 'Low';
  const riskProbability = (result.probability * 100).toFixed(1);
  const featureImportanceData = prepareFeatureImportanceData(result.feature_importance);
  const modelUsed = result.model_used || 'Advanced Model';

  // Prepare pie chart data
  const pieData = [
    { name: 'Risk Probability', value: parseFloat(riskProbability) },
    { name: 'Safe Margin', value: 100 - parseFloat(riskProbability) },
  ];

  const COLORS = ['#e91e63', '#4caf50'];

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 4, mt: 3, mb: 6 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Maternity Health Risk Assessment
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Below are the results of the risk prediction based on the provided data.
          </Typography>
        </Box>

        <Grid container spacing={4}>
          {/* Risk Summary */}
          <Grid item xs={12} md={6}>
            <Card 
              elevation={4} 
              sx={{ 
                bgcolor: riskLevel === 'High' ? 'error.light' : 'success.light',
                color: riskLevel === 'High' ? 'error.contrastText' : 'success.contrastText',
                mb: 3
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {riskLevel === 'High' ? (
                    <WarningIcon fontSize="large" sx={{ mr: 2 }} />
                  ) : (
                    <CheckCircleIcon fontSize="large" sx={{ mr: 2 }} />
                  )}
                  <Typography variant="h5" component="div">
                    {riskLevel} Risk Level
                  </Typography>
                </Box>
                <Typography variant="h3" component="div" sx={{ mb: 2 }}>
                  {riskProbability}%
                </Typography>
                <Typography variant="body1">
                  {riskLevel === 'High' 
                    ? 'Immediate medical consultation is recommended based on the prediction.'
                    : 'Regular checkups are recommended to maintain low risk status.'}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  Prediction made using: <strong>{modelUsed}</strong>
                </Typography>
              </CardContent>
            </Card>

            <Typography variant="h6" gutterBottom>
              Risk Assessment Overview
            </Typography>
            
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                  label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value}%`} />
              </PieChart>
            </ResponsiveContainer>

            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Key Contributing Factors
              </Typography>
              <List>
                {featureImportanceData.slice(0, 3).map((feature, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <PriorityHighIcon color={index === 0 ? "error" : "primary"} />
                    </ListItemIcon>
                    <ListItemText 
                      primary={feature.feature}
                      secondary={`Impact: ${(feature.value * 100).toFixed(0)}%`}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Grid>

          {/* Feature Importance Chart */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Feature Importance Analysis
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              The chart below shows how each factor contributed to the prediction.
            </Typography>
            
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={featureImportanceData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, Math.max(...featureImportanceData.map(d => d.value)) * 1.1]} />
                <YAxis type="category" dataKey="feature" />
                <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                <Bar dataKey="value" fill="#6a1b9a">
                  {featureImportanceData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={index < 3 ? '#6a1b9a' : '#9c4dcc'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Grid>

          {/* Recommendations */}
          <Grid item xs={12}>
            <Divider sx={{ my: 3 }} />
            <Typography variant="h5" gutterBottom>
              Recommendations
            </Typography>
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="body1" paragraph>
                  {riskLevel === 'High' 
                    ? 'Based on the model prediction, we recommend:'
                    : 'While your risk level is low, we still recommend:'}
                </Typography>
                <List>
                  {riskLevel === 'High' ? (
                    <>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Schedule an immediate consultation with your healthcare provider" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Discuss the specific risk factors identified in this assessment" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Consider more frequent monitoring of vital signs and symptoms" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Follow medical advice regarding lifestyle adjustments and medications" />
                      </ListItem>
                    </>
                  ) : (
                    <>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Continue with regular prenatal check-ups as scheduled" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Maintain a healthy diet and appropriate exercise routine" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Monitor for any changes in your health condition" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon><CheckCircleIcon color="primary" /></ListItemIcon>
                        <ListItemText primary="Report any concerning symptoms to your healthcare provider promptly" />
                      </ListItem>
                    </>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'space-between' }}>
          <Button 
            variant="outlined" 
            onClick={handleNewPrediction}
            startIcon={<ArrowBackIcon />}
          >
            New Prediction
          </Button>
          <Button 
            variant="contained" 
            onClick={handleDownloadPDF}
            startIcon={<FileDownloadIcon />}
          >
            Download Report
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default Results; 