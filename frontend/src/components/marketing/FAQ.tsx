/**
 * FAQ Component
 * 
 * Expandable FAQ section.
 */

import { useState } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Container,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface FAQItem {
  question: string;
  answer: string;
}

const faqs: FAQItem[] = [
  {
    question: 'How do the AI trading signals work?',
    answer:
      'Our AI models analyze multiple data sources including technical indicators, market sentiment, news analysis, and historical patterns. The system combines machine learning predictions with traditional technical analysis to generate high-confidence trading signals with comprehensive risk metrics.',
  },
  {
    question: 'Do I need trading experience to use this platform?',
    answer:
      'No, our platform is designed for traders of all experience levels. We provide clear explanations, risk metrics, and guidance for each signal. Beginners can start with conservative signals and learn as they go, while experienced traders can use advanced features and API access.',
  },
  {
    question: 'What kind of risk management features are included?',
    answer:
      'Every signal includes comprehensive risk metrics: Value at Risk (VaR), Conditional VaR (CVaR), suggested stop-loss and take-profit levels, position sizing recommendations, and risk-reward ratios. This helps you make informed decisions and manage your portfolio risk effectively.',
  },
  {
    question: 'Can I try the platform before subscribing?',
    answer:
      'Yes! We offer a free tier with limited signals so you can explore the platform. You can also start with a 7-day free trial of our premium tiers to experience all features without any credit card required.',
  },
  {
    question: 'How often are signals generated?',
    answer:
      'Signals are generated in real-time as market conditions change. You can receive notifications instantly via email or push notifications when new high-confidence signals are detected. You can also check the dashboard anytime for the latest signals.',
  },
  {
    question: 'Is my data secure?',
    answer:
      'Absolutely. We use bank-level encryption, secure authentication, and follow industry best practices for data security. Your trading data and personal information are kept confidential and secure. We never share your data with third parties.',
  },
  {
    question: 'Can I integrate signals with my trading platform?',
    answer:
      'Yes! We provide a comprehensive REST API that allows you to programmatically access signals and integrate them with your trading platform, portfolio management system, or custom trading algorithms. API access is available on our Advanced and Premium tiers.',
  },
  {
    question: 'What happens if I want to cancel?',
    answer:
      'You can cancel your subscription at any time with no penalties. Your account will remain active until the end of your billing period, and you can continue using the platform until then. No questions asked.',
  },
];

export default function FAQ() {
  const [expanded, setExpanded] = useState<string | false>(false);

  const handleChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };

  return (
    <Box sx={{ py: 8, bgcolor: 'background.default' }}>
      <Container maxWidth="md">
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography variant="h3" component="h2" gutterBottom fontWeight={700}>
            Frequently Asked Questions
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Everything you need to know about Trading Signals Pro
          </Typography>
        </Box>

        <Box>
          {faqs.map((faq, index) => (
            <Accordion
              key={index}
              expanded={expanded === `panel${index}`}
              onChange={handleChange(`panel${index}`)}
              sx={{
                mb: 2,
                '&:before': {
                  display: 'none',
                },
                boxShadow: 1,
                borderRadius: 2,
                '&.Mui-expanded': {
                  boxShadow: 2,
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                sx={{
                  py: 2,
                  '&.Mui-expanded': {
                    borderBottom: 1,
                    borderColor: 'divider',
                  },
                }}
              >
                <Typography variant="h6" fontWeight={600}>
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ py: 3 }}>
                <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      </Container>
    </Box>
  );
}
