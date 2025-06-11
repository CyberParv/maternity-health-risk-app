import React from 'react';
import { Box, Container, Typography, Link, Grid, IconButton } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import MailOutlineIcon from '@mui/icons-material/MailOutline';

function Footer() {
  return (
    <Box 
      component="footer" 
      sx={{ 
        py: 3, 
        bgcolor: 'primary.main',
        color: 'white',
        mt: 'auto'
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={3} justifyContent="space-between">
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="h6" gutterBottom>
              Maternity Health Risk Prediction
            </Typography>
            <Typography variant="body2">
              A Machine Learning-based tool to predict maternity health risks using ensemble modeling techniques.
            </Typography>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="h6" gutterBottom>
              Quick Links
            </Typography>
            <Typography variant="body2" display="block" gutterBottom>
              <Link href="/" color="inherit" underline="hover">Home</Link>
            </Typography>
            <Typography variant="body2" display="block" gutterBottom>
              <Link href="/predict" color="inherit" underline="hover">Predict Risk</Link>
            </Typography>
            <Typography variant="body2" display="block" gutterBottom>
              <Link href="/about" color="inherit" underline="hover">About the Model</Link>
            </Typography>
          </Grid>
          
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="h6" gutterBottom>
              Connect
            </Typography>
            <IconButton color="inherit" aria-label="GitHub">
              <GitHubIcon />
            </IconButton>
            <IconButton color="inherit" aria-label="LinkedIn">
              <LinkedInIcon />
            </IconButton>
            <IconButton color="inherit" aria-label="Email">
              <MailOutlineIcon />
            </IconButton>
          </Grid>
        </Grid>
        <Box mt={3}>
          <Typography variant="body2" align="center">
            {'© '}
            {new Date().getFullYear()}
            {' Maternity Health Risk Prediction. All rights reserved.'}
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}

export default Footer; 