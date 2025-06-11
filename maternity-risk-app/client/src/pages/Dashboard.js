import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Divider,
  Button,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import InsightsIcon from '@mui/icons-material/Insights';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
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
  PieChart,
  Pie,
  Cell,
} from 'recharts';

function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [timeRange, setTimeRange] = useState('month');
  const [modelFilter, setModelFilter] = useState('all');

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
  };

  const handleModelFilterChange = (event) => {
    setModelFilter(event.target.value);
  };

  // Mock data for dashboard metrics
  const overviewData = {
    totalPredictions: 1842,
    highRiskCases: 247,
    lowRiskCases: 1595,
    accuracyRate: 98.7,
    averageConfidence: 94.3,
  };

  // Mock data for risk distribution pie chart
  const riskDistributionData = [
    { name: 'High Risk', value: 247, color: '#ff5252' },
    { name: 'Low Risk', value: 1595, color: '#4caf50' },
  ];

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

  // Mock data for monthly predictions trend
  const monthlyTrendData = [
    { month: 'Jan', predictions: 120, highRisk: 16 },
    { month: 'Feb', predictions: 145, highRisk: 19 },
    { month: 'Mar', predictions: 162, highRisk: 22 },
    { month: 'Apr', predictions: 178, highRisk: 24 },
    { month: 'May', predictions: 189, highRisk: 25 },
    { month: 'Jun', predictions: 201, highRisk: 27 },
    { month: 'Jul', predictions: 210, highRisk: 29 },
    { month: 'Aug', predictions: 198, highRisk: 27 },
    { month: 'Sep', predictions: 220, highRisk: 30 },
    { month: 'Oct', predictions: 232, highRisk: 32 },
    { month: 'Nov', predictions: 245, highRisk: 35 },
    { month: 'Dec', predictions: 237, highRisk: 33 },
  ];

  // Weekly data for more granular view
  const weeklyTrendData = [
    { week: 'Week 1', predictions: 52, highRisk: 7 },
    { week: 'Week 2', predictions: 58, highRisk: 8 },
    { week: 'Week 3', predictions: 63, highRisk: 9 },
    { week: 'Week 4', predictions: 72, highRisk: 11 },
    { week: 'Week 5', predictions: 67, highRisk: 9 },
    { week: 'Week 6', predictions: 59, highRisk: 8 },
  ];

  // Function to get appropriate trend data based on selected time range
  const getTrendData = () => {
    if (timeRange === 'week') return weeklyTrendData;
    return monthlyTrendData;
  };

  // X-axis key for trend chart based on time range
  const trendXAxisKey = timeRange === 'week' ? 'week' : 'month';

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 3, mt: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <DashboardIcon color="primary" sx={{ fontSize: 28, mr: 2 }} />
          <Typography variant="h4" component="h1">
            Maternity Risk Dashboard
          </Typography>
        </Box>

        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          sx={{ mb: 3 }}
        >
          <Tab icon={<AnalyticsIcon />} label="Overview" />
          <Tab icon={<InsightsIcon />} label="Model Performance" />
          <Tab icon={<TrendingUpIcon />} label="Trend Analysis" />
        </Tabs>

        {/* OVERVIEW TAB */}
        {activeTab === 0 && (
          <Box>
            {/* Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={4}>
                <Card sx={{ height: '100%', bgcolor: '#f5f5f5' }}>
                  <CardContent>
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Total Predictions
                    </Typography>
                    <Typography variant="h3" color="primary">
                      {overviewData.totalPredictions}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      All-time predictions made by the system
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <Card sx={{ height: '100%', bgcolor: '#fff8f8' }}>
                  <CardContent>
                    <Typography variant="h6" color="error" gutterBottom>
                      High Risk Cases
                    </Typography>
                    <Typography variant="h3" color="error">
                      {overviewData.highRiskCases}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      {(overviewData.highRiskCases / overviewData.totalPredictions * 100).toFixed(1)}% of total cases
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <Card sx={{ height: '100%', bgcolor: '#f8fff8' }}>
                  <CardContent>
                    <Typography variant="h6" color="success" gutterBottom>
                      Model Accuracy
                    </Typography>
                    <Typography variant="h3" color="success">
                      {overviewData.accuracyRate}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Based on validated patient outcomes
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Risk Distribution and Feature Importance */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={5}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Risk Level Distribution
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={riskDistributionData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            outerRadius={100}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            {riskDistributionData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={7}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Feature Importance
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={featureImportance}
                          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" domain={[0, 0.25]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                          <YAxis dataKey="feature" type="category" width={80} />
                          <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                          <Bar dataKey="importance" fill="#8884d8" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Images Section */}
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Model Visualization Gallery
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        Model Comparison
                      </Typography>
                      <Box
                        component="img"
                        src="/visualizations/model_comparison.png"
                        alt="Model Comparison"
                        sx={{ width: '100%', height: 'auto', borderRadius: 1 }}
                      />
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        Confusion Matrices
                      </Typography>
                      <Box
                        component="img"
                        src="/visualizations/confusion_matrices.png"
                        alt="Confusion Matrices"
                        sx={{ width: '100%', height: 'auto', borderRadius: 1 }}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          </Box>
        )}

        {/* MODEL PERFORMANCE TAB */}
        {activeTab === 1 && (
          <Box>
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <FormControl sx={{ width: 200 }}>
                <InputLabel>Model Filter</InputLabel>
                <Select
                  value={modelFilter}
                  onChange={handleModelFilterChange}
                  label="Model Filter"
                >
                  <MenuItem value="all">All Models</MenuItem>
                  <MenuItem value="ensemble">Ensemble Only</MenuItem>
                  <MenuItem value="individual">Individual Models</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Performance Metrics
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={modelPerformance.filter(model => {
                        if (modelFilter === 'all') return true;
                        if (modelFilter === 'ensemble') return model.name.includes('Ensemble') || model.name.includes('Voting');
                        if (modelFilter === 'individual') return !model.name.includes('Ensemble') && !model.name.includes('Voting');
                        return true;
                      })}
                      margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} />
                      <YAxis domain={[80, 100]} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="accuracy" name="Accuracy (%)" fill="#8884d8" />
                      <Bar dataKey="precision" name="Precision (%)" fill="#82ca9d" />
                      <Bar dataKey="recall" name="Recall (%)" fill="#ffc658" />
                      <Bar dataKey="f1" name="F1 Score (%)" fill="#ff8042" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>

            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Feature Importance
                    </Typography>
                    <Box
                      component="img"
                      src="/visualizations/feature_importance.png"
                      alt="Feature Importance"
                      sx={{ width: '100%', height: 'auto', borderRadius: 1 }}
                    />
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Correlation with Risk Level
                    </Typography>
                    <Box
                      component="img"
                      src="/visualizations/correlation_with_target.png"
                      alt="Correlation With Target"
                      sx={{ width: '100%', height: 'auto', borderRadius: 1 }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* TREND ANALYSIS TAB */}
        {activeTab === 2 && (
          <Box>
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <FormControl sx={{ width: 150 }}>
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  onChange={handleTimeRangeChange}
                  label="Time Range"
                >
                  <MenuItem value="month">Monthly</MenuItem>
                  <MenuItem value="week">Weekly</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Prediction Trends
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={getTrendData()}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey={trendXAxisKey} />
                      <YAxis yAxisId="left" orientation="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="predictions"
                        name="Total Predictions"
                        stroke="#8884d8"
                        activeDot={{ r: 8 }}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="highRisk"
                        name="High Risk Cases"
                        stroke="#ff5252"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Risk Rate Trend
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={getTrendData().map(item => ({
                        ...item,
                        riskRate: (item.highRisk / item.predictions * 100).toFixed(1)
                      }))}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey={trendXAxisKey} />
                      <YAxis domain={[0, 20]} tickFormatter={(value) => `${value}%`} />
                      <Tooltip formatter={(value) => `${value}%`} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="riskRate"
                        name="High Risk Rate"
                        stroke="#ff8042"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default Dashboard; 