import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData, Time, LineSeries } from 'lightweight-charts';
import { Box, Typography } from '@mui/material';

interface Prediction {
  period: string;
  direction: 'UP' | 'DOWN';
  confidence: number;
}

interface PredictionChartProps {
  predictions: Prediction[];
  currentPrice: number;
  height?: number;
}

export default function PredictionChart({ predictions, currentPrice, height = 100 }: PredictionChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const lineSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) {
      return;
    }

    if (!predictions || predictions.length === 0) {
      setError('No predictions available');
      return;
    }

    // Wait for container to have dimensions
    const checkDimensions = () => {
      if (chartContainerRef.current && chartContainerRef.current.clientWidth > 0) {
        initializeChart();
      } else {
        // Retry after a short delay
        setTimeout(checkDimensions, 100);
      }
    };

    const initializeChart = () => {
      try {
        // Clean up existing chart
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }

        if (!chartContainerRef.current) return;

        const container = chartContainerRef.current;
        const width = container.clientWidth || 400;
        const chartHeight = height;

        // Create chart
        const chart = createChart(container, {
          layout: {
            background: { color: 'transparent' },
            textColor: '#94a3b8',
          },
          grid: {
            vertLines: { visible: false },
            horzLines: { visible: false },
          },
          width: width,
          height: chartHeight,
          timeScale: {
            visible: false,
          },
          rightPriceScale: {
            visible: false,
          },
          leftPriceScale: {
            visible: false,
          },
        });

        // Generate data points from predictions
        const lineData: LineData[] = [];
        const upperLineData: LineData[] = [];
        const lowerLineData: LineData[] = [];

        // Start from current price
        let price = currentPrice;
        const baseTime = Date.now() / 1000;

        // Add starting point
        const startTime = baseTime as Time;
        lineData.push({ time: startTime, value: currentPrice });
        upperLineData.push({ time: startTime, value: currentPrice * 1.01 });
        lowerLineData.push({ time: startTime, value: currentPrice * 0.99 });

        predictions.forEach((pred) => {
          const time = (baseTime + (predictions.indexOf(pred) + 1) * 3600) as Time; // Each hour

          // Calculate price movement based on direction and confidence
          const movementPercent = (pred.confidence / 100) * 0.015; // 1.5% max movement
          const movement = pred.direction === 'UP'
            ? price * movementPercent
            : -price * movementPercent;

          price += movement;

          // Confidence interval bounds (wider for lower confidence)
          const confidenceRange = (100 - pred.confidence) / 100;
          const upperBound = price * (1 + confidenceRange * 0.015);
          const lowerBound = price * (1 - confidenceRange * 0.015);

          lineData.push({ time, value: price });
          upperLineData.push({ time, value: upperBound });
          lowerLineData.push({ time, value: lowerBound });
        });

        // Add upper confidence line using v5 API
        const upperLine = chart.addSeries(LineSeries, {
          color: 'rgba(139, 92, 246, 0.3)',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          priceLineVisible: false,
          lastValueVisible: false,
        });

        // Add lower confidence line using v5 API
        const lowerLine = chart.addSeries(LineSeries, {
          color: 'rgba(139, 92, 246, 0.3)',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          priceLineVisible: false,
          lastValueVisible: false,
        });

        // Add main prediction line using v5 API
        const lineSeries = chart.addSeries(LineSeries, {
          color: '#8b5cf6',
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        // Set data
        lineSeries.setData(lineData);
        upperLine.setData(upperLineData);
        lowerLine.setData(lowerLineData);

        // Store references
        chartRef.current = chart;
        lineSeriesRef.current = lineSeries;
        setError(null);

        // Handle resize
        const handleResize = () => {
          if (chartContainerRef.current && chartRef.current) {
            const newWidth = chartContainerRef.current.clientWidth;
            if (newWidth > 0) {
              chartRef.current.applyOptions({
                width: newWidth,
              });
            }
          }
        };

        window.addEventListener('resize', handleResize);

        // Cleanup function
        return () => {
          window.removeEventListener('resize', handleResize);
          if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
          }
        };
      } catch (err: any) {
        console.error('Error creating prediction chart:', err);
        setError(err.message || 'Failed to load chart');
      }
    };

    // Start initialization
    checkDimensions();
  }, [predictions, currentPrice, height]);

  if (error) {
    return (
      <Box sx={{
        height: `${height}px`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#64748b',
        flexDirection: 'column',
        gap: 1,
      }}>
        <Typography variant="caption" sx={{ fontSize: '0.85rem' }}>{error}</Typography>
      </Box>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <Box sx={{
        height: `${height}px`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#64748b',
      }}>
        <Typography sx={{ fontSize: '0.85rem' }}>Loading predictions...</Typography>
      </Box>
    );
  }

  return (
    <div
      ref={chartContainerRef}
      style={{
        width: '100%',
        height: `${height}px`,
        position: 'relative',
        minWidth: '200px',
      }}
    />
  );
}

