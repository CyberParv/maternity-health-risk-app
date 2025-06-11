import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  Grid,
  Button,
  MenuItem,
  FormControl,
  FormControlLabel,
  FormLabel,
  RadioGroup,
  Radio,
  Slider,
  InputAdornment,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Card,
  CardContent,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { predictRisk } from '../services/api';

const DEFAULT_FORM_VALUES = {
  systolicBP: '',
  diastolicBP: '',
  bs: '',
  bodyTemp: '37',  // Default normal body temperature
  bmi: '',
  heartRate: '',
  previousComplications: '0',
  preexistingDiabetes: '0',
  gestationalDiabetes: '0',
  mentalHealth: '0',
};

const INPUT_TOOLTIPS = {
  systolicBP: 'Upper blood pressure reading in mmHg. Normal range: 90-120',
  diastolicBP: 'Lower blood pressure reading in mmHg. Normal range: 60-80',
  bs: 'Blood sugar level in mmol/L. Normal: 4.0-5.4 (fasting)',
  bodyTemp: 'Body temperature in Celsius. Normal: 36.5-37.5',
  bmi: 'Body Mass Index = weight(kg) / height²(m). Normal: 18.5-24.9',
  heartRate: 'Heart rate in beats per minute. Normal range: 60-100',
  previousComplications: 'Complications during previous pregnancies',
  preexistingDiabetes: 'Diabetes diagnosed before pregnancy', 
  gestationalDiabetes: 'Diabetes that develops during pregnancy',
  mentalHealth: 'Mental health conditions like depression, anxiety, etc.'
};

const steps = [
  {
    label: 'Clinical Measurements',
    description: 'Enter basic clinical measurements',
    fields: ['systolicBP', 'diastolicBP', 'bs', 'bodyTemp', 'heartRate']
  },
  {
    label: 'Body Metrics',
    description: 'Enter body measurements',
    fields: ['bmi']
  },
  {
    label: 'Medical History',
    description: 'Enter information about medical history',
    fields: ['previousComplications', 'preexistingDiabetes', 'gestationalDiabetes', 'mentalHealth']
  },
  {
    label: 'Review',
    description: 'Review your information before submission',
    fields: []
  }
];

function PredictionForm() {
  const [formValues, setFormValues] = useState(DEFAULT_FORM_VALUES);
  const [activeStep, setActiveStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: value,
    });
  };

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setFormValues(DEFAULT_FORM_VALUES);
    setActiveStep(0);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      // Transform form data to match the expected model input
      const modelInput = {
        systolic_bp: parseFloat(formValues.systolicBP),
        diastolic: parseFloat(formValues.diastolicBP),
        bs: parseFloat(formValues.bs),
        bmi: parseFloat(formValues.bmi),
        heart_rate: parseFloat(formValues.heartRate),
        previous_complications: parseInt(formValues.previousComplications),
        preexisting_diabetes: parseInt(formValues.preexistingDiabetes),
        gestational_diabetes: parseInt(formValues.gestationalDiabetes),
        mental_health: parseInt(formValues.mentalHealth),
      };

      // Call the API for prediction using our service
      const response = await predictRisk(modelInput);
      
      // Store result in session storage for results page
      sessionStorage.setItem('predictionResult', JSON.stringify(response));
      sessionStorage.setItem('patientData', JSON.stringify(formValues));
      
      // Navigate to results page
      navigate('/results');
      setIsSubmitting(false);

    } catch (err) {
      setError('An error occurred while submitting your data. Please try again.');
      setIsSubmitting(false);
      console.error('Error submitting form:', err);
    }
  };

  const validateStep = (step) => {
    const currentFields = steps[step].fields;
    for (const field of currentFields) {
      if (!formValues[field] || formValues[field] === '') {
        return false;
      }
    }
    return true;
  };

  const renderTextField = (field, label, type = 'number') => (
    <Grid item xs={12} sm={6} md={4} key={field}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <TextField
          fullWidth
          label={label}
          name={field}
          type={type}
          value={formValues[field]}
          onChange={handleChange}
          required
          variant="outlined"
          margin="normal"
          InputProps={{
            endAdornment: field === 'systolicBP' || field === 'diastolicBP' ? (
              <InputAdornment position="end">mmHg</InputAdornment>
            ) : field === 'bs' ? (
              <InputAdornment position="end">mmol/L</InputAdornment>
            ) : field === 'bodyTemp' ? (
              <InputAdornment position="end">°C</InputAdornment>
            ) : field === 'heartRate' ? (
              <InputAdornment position="end">bpm</InputAdornment>
            ) : field === 'bmi' ? (
              <InputAdornment position="end">kg/m²</InputAdornment>
            ) : null
          }}
        />
        <Tooltip title={INPUT_TOOLTIPS[field] || ''}>
          <IconButton>
            <HelpOutlineIcon />
          </IconButton>
        </Tooltip>
      </Box>
    </Grid>
  );

  const renderRadioField = (field, label) => (
    <Grid item xs={12} sm={6} key={field}>
      <FormControl component="fieldset" margin="normal">
        <FormLabel component="legend">{label}</FormLabel>
        <RadioGroup
          row
          name={field}
          value={formValues[field]}
          onChange={handleChange}
        >
          <FormControlLabel value="0" control={<Radio />} label="No" />
          <FormControlLabel value="1" control={<Radio />} label="Yes" />
        </RadioGroup>
        <Tooltip title={INPUT_TOOLTIPS[field] || ''}>
          <IconButton size="small">
            <HelpOutlineIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </FormControl>
    </Grid>
  );

  const renderReviewCard = (title, fields) => {
    return (
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>{title}</Typography>
          <Grid container spacing={2}>
            {fields.map(({ name, label, unit }) => (
              <Grid item xs={6} sm={4} key={name}>
                <Typography variant="subtitle2" color="text.secondary">{label}</Typography>
                <Typography variant="body1">
                  {formValues[name]} {unit || ''}
                </Typography>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    );
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0: // Clinical Measurements
        return (
          <Grid container spacing={2}>
            {renderTextField('systolicBP', 'Systolic Blood Pressure')}
            {renderTextField('diastolicBP', 'Diastolic Blood Pressure')}
            {renderTextField('bs', 'Blood Sugar')}
            {renderTextField('bodyTemp', 'Body Temperature')}
            {renderTextField('heartRate', 'Heart Rate')}
          </Grid>
        );
      case 1: // Body Metrics
        return (
          <Grid container spacing={2}>
            {renderTextField('bmi', 'Body Mass Index (BMI)')}
          </Grid>
        );
      case 2: // Medical History
        return (
          <Grid container spacing={2}>
            {renderRadioField('previousComplications', 'Previous Pregnancy Complications')}
            {renderRadioField('preexistingDiabetes', 'Preexisting Diabetes')}
            {renderRadioField('gestationalDiabetes', 'Gestational Diabetes')}
            {renderRadioField('mentalHealth', 'Mental Health Condition')}
          </Grid>
        );
      case 3: // Review
        return (
          <Box>
            <Typography variant="h6" gutterBottom>Review Your Information</Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Please review the information below before submitting. You can go back to edit if needed.
            </Typography>
            
            {renderReviewCard('Clinical Measurements', [
              { name: 'systolicBP', label: 'Systolic BP', unit: 'mmHg' },
              { name: 'diastolicBP', label: 'Diastolic BP', unit: 'mmHg' },
              { name: 'bs', label: 'Blood Sugar', unit: 'mmol/L' },
              { name: 'bodyTemp', label: 'Body Temperature', unit: '°C' },
              { name: 'heartRate', label: 'Heart Rate', unit: 'bpm' },
            ])}
            
            {renderReviewCard('Body Metrics', [
              { name: 'bmi', label: 'BMI', unit: 'kg/m²' },
            ])}
            
            {renderReviewCard('Medical History', [
              { name: 'previousComplications', label: 'Previous Complications', unit: formValues.previousComplications === '1' ? 'Yes' : 'No' },
              { name: 'preexistingDiabetes', label: 'Preexisting Diabetes', unit: formValues.preexistingDiabetes === '1' ? 'Yes' : 'No' },
              { name: 'gestationalDiabetes', label: 'Gestational Diabetes', unit: formValues.gestationalDiabetes === '1' ? 'Yes' : 'No' },
              { name: 'mentalHealth', label: 'Mental Health Condition', unit: formValues.mentalHealth === '1' ? 'Yes' : 'No' },
            ])}
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 3, mb: 8 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Maternity Health Risk Prediction
          </Typography>
        </Box>
        
        <Typography variant="body1" paragraph>
          Enter the required information to predict maternity health risk. All fields are required for accurate prediction.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        <form onSubmit={handleSubmit}>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.label}>
                <StepLabel>{step.label}</StepLabel>
                <StepContent>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {step.description}
                  </Typography>
                  {renderStepContent(index)}
                  <Box sx={{ mb: 2, mt: 3 }}>
                    <div>
                      <Button
                        disabled={activeStep === 0}
                        onClick={handleBack}
                        sx={{ mr: 1 }}
                        startIcon={<ArrowBackIcon />}
                      >
                        Back
                      </Button>
                      {activeStep === steps.length - 1 ? (
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={handleSubmit}
                          disabled={isSubmitting}
                        >
                          {isSubmitting ? 'Submitting...' : 'Submit'}
                        </Button>
                      ) : (
                        <Button
                          variant="contained"
                          onClick={handleNext}
                          disabled={!validateStep(activeStep)}
                          endIcon={<ArrowForwardIcon />}
                        >
                          Next
                        </Button>
                      )}
                    </div>
                  </Box>
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </form>
      </Paper>
    </Container>
  );
}

export default PredictionForm; 