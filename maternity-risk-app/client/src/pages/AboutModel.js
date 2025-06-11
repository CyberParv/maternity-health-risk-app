import React from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
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
  LineChart,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import InfoIcon from '@mui/icons-material/Info';
import BarChartIcon from '@mui/icons-material/BarChart';
import TimelineIcon from '@mui/icons-material/Timeline';

function AboutModel() {
  // Mock data for model performance metrics
  const modelPerformance = [
    { name: 'Random Forest', accuracy: 98.0, precision: 100.0, recall: 90.62, f1: 95.08 },
    { name: 'Neural Network', accuracy: 97.33, precision: 100.0, recall: 87.5, f1: 93.33 },
    { name: 'Stack Ensemble', accuracy: 97.33, precision: 100.0, recall: 87.5, f1: 93.33 },
    { name: 'Weighted Voting', accuracy: 96.67, precision: 90.91, recall: 93.75, f1: 92.31 },
    { name: 'Logistic Regression', accuracy: 95.33, precision: 86.36, recall: 90.38, f1: 88.22 },
  ];

  // Mock data for feature importance
  const featureImportance = [
    { feature: 'Preexisting Diabetes', importance: 0.24 },
    { feature: 'Mental Health', importance: 0.18 },
    { feature: 'Gestational Diabetes', importance: 0.16 },
    { feature: 'Previous Complications', importance: 0.15 },
    { feature: 'Systolic BP', importance: 0.12 },
    { feature: 'BMI', importance: 0.09 },
    { feature: 'Diastolic BP', importance: 0.06 },
  ];

  // Mock data for cross-validation results
  const crossValidationData = [
    { fold: 'Fold 1', accuracy: 96.8 },
    { fold: 'Fold 2', accuracy: 97.5 },
    { fold: 'Fold 3', accuracy: 98.1 },
    { fold: 'Fold 4', accuracy: 96.9 },
    { fold: 'Fold 5', accuracy: 97.8 },
  ];

  // Mock data for model radar chart
  const radarData = [
    {
      model: 'Random Forest',
      accuracy: 98.0,
      precision: 100.0,
      recall: 90.62,
      f1: 95.08,
      auc: 97.5,
    },
    {
      model: 'Neural Network',
      accuracy: 97.33,
      precision: 100.0,
      recall: 87.5,
      f1: 93.33,
      auc: 96.8,
    },
    {
      model: 'Ensemble Model',
      accuracy: 97.33,
      precision: 100.0,
      recall: 87.5,
      f1: 93.33,
      auc: 97.2,
    },
  ];

  // Normalize data for radar chart (0-100 scale)
  const normalizedRadarData = radarData.map(item => ({
    subject: item.model,
    accuracy: item.accuracy,
    precision: item.precision,
    recall: item.recall,
    f1: item.f1,
    auc: item.auc,
  }));

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 4, mt: 3, mb: 6 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            About the Maternity Health Risk Model
          </Typography>
          <Typography variant="body1" color="text.secondary">
            This page provides information about the machine learning model used for
            maternity health risk prediction, its performance metrics, and feature importance.
          </Typography>
        </Box>

        <Grid container spacing={4}>
          {/* Model Overview */}
          <Grid item xs={12}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Model Overview
                </Typography>
                <Typography variant="body1" paragraph>
                  Our maternity health risk prediction system uses an ensemble of machine learning models
                  to provide accurate risk assessments. The system combines the strengths of multiple algorithms,
                  including Random Forest, Neural Networks, and Logistic Regression, to achieve optimal performance.
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <List>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircleOutlineIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Ensemble Approach" 
                          secondary="Combines predictions from multiple models for improved accuracy"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircleOutlineIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Cross-Validation" 
                          secondary="Validated across multiple data splits to ensure robustness"
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <List>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircleOutlineIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Feature Importance Analysis" 
                          secondary="Identifies key factors influencing risk predictions"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircleOutlineIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Optimized Thresholds" 
                          secondary="Calibrated for balanced precision and recall in medical context"
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Model Performance */}
          <Grid item xs={12} md={7}>
            <Typography variant="h5" gutterBottom>
              <BarChartIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Model Performance Comparison
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Comparison of performance metrics across different models used in our ensemble.
            </Typography>
            
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={modelPerformance}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} />
                <YAxis domain={[80, 100]} />
                <Tooltip formatter={(value) => `${value}%`} />
                <Legend />
                <Bar dataKey="accuracy" name="Accuracy" fill="#6a1b9a" />
                <Bar dataKey="precision" name="Precision" fill="#1976d2" />
                <Bar dataKey="recall" name="Recall" fill="#388e3c" />
                <Bar dataKey="f1" name="F1 Score" fill="#e53935" />
              </BarChart>
            </ResponsiveContainer>
            
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Key Performance Insights
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <InfoIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Random Forest is the top performing individual model"
                    secondary="With 98% accuracy and 100% precision"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <InfoIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Ensemble models provide robust performance"
                    secondary="Offering a balance between precision and recall"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <InfoIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="All models show high specificity"
                    secondary="Minimizing false positives, which is crucial in medical applications"
                  />
                </ListItem>
              </List>
            </Box>
          </Grid>

          {/* Radar Chart */}
          <Grid item xs={12} md={5}>
            <Typography variant="h5" gutterBottom>
              <TimelineIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Multi-Metric Evaluation
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Radar visualization of top models across all performance metrics.
            </Typography>
            
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart outerRadius={150} data={normalizedRadarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={30} domain={[80, 100]} />
                <Radar name="Random Forest" dataKey="accuracy" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                <Radar name="Neural Network" dataKey="precision" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                <Radar name="Ensemble Model" dataKey="recall" stroke="#ffc658" fill="#ffc658" fillOpacity={0.6} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
            
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Cross-Validation Results
              </Typography>
              <Typography variant="body2" paragraph>
                5-fold cross-validation ensures the model's consistent performance across different data splits.
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={crossValidationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fold" />
                  <YAxis domain={[95, 100]} />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#6a1b9a" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          {/* Feature Importance */}
          <Grid item xs={12}>
            <Divider sx={{ my: 3 }} />
            <Typography variant="h5" gutterBottom>
              Feature Importance
            </Typography>
            <Typography variant="body1" paragraph>
              The chart below shows the relative importance of each feature in the prediction model.
              These values represent how much each factor contributes to the final risk assessment.
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={featureImportance}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="feature" />
                <YAxis domain={[0, 0.3]} />
                <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                <Legend />
                <Bar dataKey="importance" name="Importance Score" fill="#6a1b9a" />
              </BarChart>
            </ResponsiveContainer>
            
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Feature Importance Insights
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Key Predictors
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Preexisting Diabetes is the most influential factor"
                            secondary="Contributing approximately 24% to the prediction"
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Mental Health is the second most important factor"
                            secondary="Contributing approximately 18% to the prediction"
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Gestational Diabetes and Previous Complications"
                            secondary="Together contribute about 31% to the prediction"
                          />
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Clinical Implications
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Medical History Factors"
                            secondary="Dominate the top of the importance list, suggesting preexisting conditions are highly predictive"
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Vital Signs"
                            secondary="Blood pressure measurements and BMI contribute significantly to risk prediction"
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Holistic Assessment"
                            secondary="The model considers a balanced mix of both clinical measurements and medical history"
                          />
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          </Grid>

          {/* Model Development and Validation */}
          <Grid item xs={12}>
            <Divider sx={{ my: 3 }} />
            <Typography variant="h5" gutterBottom>
              Model Development and Validation
            </Typography>
            <Typography variant="body1" paragraph>
              Our model was developed using a rigorous methodology to ensure reliability and accuracy:
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Data Processing
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText primary="Data preprocessing included normalization of numerical features" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Missing values were handled with advanced imputation techniques" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Dataset was split into training (70%), validation (15%), and test (15%) sets" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Class imbalance was addressed with appropriate resampling methods" />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Model Training
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText primary="Multiple algorithms were trained and evaluated" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Hyperparameter optimization was performed using grid search" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="K-fold cross-validation (k=5) was used to assess model stability" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="Ensemble methods were applied to combine individual model strengths" />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Evaluation Metrics
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Accuracy" 
                          secondary="Overall correctness of predictions (98% for best model)"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Precision" 
                          secondary="Ratio of correct positive predictions to all positive predictions"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Recall (Sensitivity)" 
                          secondary="Ability to identify all positive cases"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="F1-Score" 
                          secondary="Harmonic mean of precision and recall"
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
}

export default AboutModel; 