/**
 * Onboarding Flow Component
 * 
 * Guided onboarding for new users after registration.
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Card,
  CardContent,
  Container,
} from '@mui/material';
import WelcomeStep from './WelcomeStep';
import TradingStyleQuiz from './TradingStyleQuiz';
import WatchlistSetup from './WatchlistSetup';
import DashboardTour from './DashboardTour';
import FirstActionPrompt from './FirstActionPrompt';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiClient from '../../services/api';

const steps = ['Welcome', 'Trading Style', 'Watchlist', 'Tour', 'Get Started'];

interface OnboardingData {
  risk_tolerance?: string;
  preferred_sectors?: string[];
  trading_experience?: string;
  primary_goal?: string;
  check_frequency?: string;
  watchlist?: string[];
}

export default function OnboardingFlow() {
  const [activeStep, setActiveStep] = useState(0);
  const [data, setData] = useState<OnboardingData>({});
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: preferences } = useQuery({
    queryKey: ['user-preferences'],
    queryFn: async () => {
      const response = await apiClient.get('/account/preferences');
      return response.data.data;
    },
  });

  const savePreferencesMutation = useMutation({
    mutationFn: async (prefs: any) => {
      const response = await apiClient.post('/account/preferences', prefs);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user-preferences'] });
    },
  });

  const completeOnboardingMutation = useMutation({
    mutationFn: async () => {
      const response = await apiClient.post('/account/onboarding/complete');
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user'] });
      navigate('/dashboard');
    },
  });

  const handleNext = (stepData?: Partial<OnboardingData>) => {
    if (stepData) {
      setData({ ...data, ...stepData });
    }

    if (activeStep === steps.length - 1) {
      // Save all preferences and complete onboarding
      savePreferencesMutation.mutate({
        ...data,
        ...stepData,
        onboarding_completed: true,
      });
      completeOnboardingMutation.mutate();
    } else {
      setActiveStep((prev) => prev + 1);
    }
  };

  const handleSkip = () => {
    completeOnboardingMutation.mutate();
  };

  const renderStep = () => {
    switch (activeStep) {
      case 0:
        return <WelcomeStep onNext={() => handleNext()} onSkip={handleSkip} />;
      case 1:
        return (
          <TradingStyleQuiz
            onNext={(quizData) => handleNext(quizData)}
            onSkip={handleSkip}
          />
        );
      case 2:
        return (
          <WatchlistSetup
            onNext={(watchlist) => handleNext({ watchlist })}
            onSkip={handleSkip}
          />
        );
      case 3:
        return <DashboardTour onNext={() => handleNext()} onSkip={handleSkip} />;
      case 4:
        return <FirstActionPrompt onComplete={handleSkip} />;
      default:
        return null;
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ py: 4 }}>
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Card>
          <CardContent sx={{ p: 4, minHeight: 400 }}>
            {renderStep()}
          </CardContent>
        </Card>

        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Step {activeStep + 1} of {steps.length}
          </Typography>
        </Box>
      </Box>
    </Container>
  );
}
