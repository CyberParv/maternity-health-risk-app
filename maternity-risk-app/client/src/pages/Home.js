import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Grid,
  Typography,
  Card,
  CardContent,
  CardMedia,
  Stack,
  Paper,
} from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import GroupsIcon from '@mui/icons-material/Groups';
import InsightsIcon from '@mui/icons-material/Insights';

function Home() {
  const navigate = useNavigate();

  return (
    <Box>
      {/* Hero Section */}
      <Box 
        sx={{
          background: 'linear-gradient(to right, #6a1b9a, #9c4dcc)',
          color: 'white',
          py: 8,
          borderRadius: { xs: 0, md: '0 0 20px 20px' },
          mb: 6
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h2" component="h1" gutterBottom>
                Maternity Health Risk Prediction
              </Typography>
              <Typography variant="h5" paragraph>
                Predict and prevent pregnancy complications with our advanced AI-powered platform.
              </Typography>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} mt={4}>
                <Button 
                  variant="contained" 
                  size="large" 
                  color="secondary" 
                  onClick={() => navigate('/predict')}
                >
                  Start Prediction
                </Button>
                <Button 
                  variant="outlined" 
                  size="large" 
                  sx={{ color: 'white', borderColor: 'white' }}
                  onClick={() => navigate('/about')}
                >
                  Learn More
                </Button>
              </Stack>
            </Grid>
            <Grid item xs={12} md={6} sx={{ display: { xs: 'none', md: 'block' } }}>
              <Box 
                component="img"
                src="https://img.freepik.com/free-photo/young-pregnant-woman-talking-with-doctor_23-2149344214.jpg"
                alt="Maternity care"
                sx={{ 
                  width: '100%', 
                  maxHeight: 400, 
                  objectFit: 'cover',
                  borderRadius: 4,
                  boxShadow: 5
                }}
              />
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ mb: 8 }}>
        <Typography variant="h3" component="h2" align="center" gutterBottom>
          Why Use Our Platform?
        </Typography>
        <Typography variant="h6" align="center" color="text.secondary" paragraph sx={{ mb: 6 }}>
          Our machine learning model is designed to help healthcare professionals predict maternity health risks.
        </Typography>

        <Grid container spacing={4}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                <AssessmentIcon color="primary" sx={{ fontSize: 60 }} />
              </Box>
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h5" component="h3" align="center">
                  Accurate Predictions
                </Typography>
                <Typography align="center">
                  98.67% accuracy using our enhanced ensemble model trained on comprehensive maternity data.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                <HealthAndSafetyIcon color="primary" sx={{ fontSize: 60 }} />
              </Box>
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h5" component="h3" align="center">
                  Risk Prevention
                </Typography>
                <Typography align="center">
                  Early identification of high-risk pregnancies allows for timely interventions.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                <GroupsIcon color="primary" sx={{ fontSize: 60 }} />
              </Box>
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h5" component="h3" align="center">
                  Better Care
                </Typography>
                <Typography align="center">
                  Personalized care recommendations based on individual risk profiles.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                <InsightsIcon color="primary" sx={{ fontSize: 60 }} />
              </Box>
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h5" component="h3" align="center">
                  Data Insights
                </Typography>
                <Typography align="center">
                  Advanced analytics help identify the most important risk factors for each patient.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Call to Action */}
      <Box sx={{ bgcolor: 'secondary.light', py: 6, mb: 8 }}>
        <Container maxWidth="md">
          <Typography variant="h4" align="center" gutterBottom>
            Ready to Assess Maternity Health Risks?
          </Typography>
          <Typography variant="h6" align="center" color="text.secondary" paragraph>
            Make informed decisions with our advanced risk prediction model.
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <Button 
              variant="contained" 
              color="primary" 
              size="large" 
              onClick={() => navigate('/predict')}
            >
              Start Your Prediction Now
            </Button>
          </Box>
        </Container>
      </Box>
    </Box>
  );
}

export default Home; 