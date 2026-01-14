import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, CandlestickSeries } from 'lightweight-charts';
import { Box, Typography } from '@mui/material';

interface CandlestickChartProps {
  data: CandlestickData[];
  height?: number;
  symbol?: string;
}

export default function CandlestickChart({ data, height = 280 }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!chartContainerRef.current) {
      return;
    }

    if (!data || data.length === 0) {
      setError('No data available');
      return;
    }

    // Wait for container to have dimensions
    const checkDimensions = () => {
      if (chartContainerRef.current && chartContainerRef.current.clientWidth > 0) {
        initializeChart();
      } else {
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
        const width = container.clientWidth || 800;
        const chartHeight = height;

        // Calculate optimal bar spacing based on data length
        const dataLength = data.length;
        const availableWidth = width - 60; // Reserve space for price scale
        const optimalBarSpacing = Math.max(6, Math.min(15, availableWidth / dataLength));

        // Create chart with better initial display settings
        const chart = createChart(container, {
          layout: {
            background: { color: 'transparent' },
            textColor: '#94a3b8',
            fontFamily: 'Inter, system-ui, sans-serif',
          },
          grid: {
            vertLines: { color: 'rgba(148, 163, 184, 0.08)' },
            horzLines: { color: 'rgba(148, 163, 184, 0.08)' },
          },
          width: width,
          height: chartHeight,
          timeScale: {
            timeVisible: true,
            secondsVisible: false,
            borderColor: 'rgba(148, 163, 184, 0.15)',
            barSpacing: optimalBarSpacing,
            rightOffset: 3,
            minBarSpacing: 4,
            fixLeftEdge: true,
            fixRightEdge: true,
          },
          rightPriceScale: {
            borderColor: 'rgba(148, 163, 184, 0.15)',
            scaleMargins: {
              top: 0.08,
              bottom: 0.08,
            },
            autoScale: true,
          },
          crosshair: {
            vertLine: {
              color: 'rgba(59, 130, 246, 0.5)',
              labelBackgroundColor: '#3b82f6',
            },
            horzLine: {
              color: 'rgba(59, 130, 246, 0.5)',
              labelBackgroundColor: '#3b82f6',
            },
          },
          handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: false,
          },
          handleScale: {
            mouseWheel: true,
            pinch: true,
            axisPressedMouseMove: true,
          },
        });

        // Add candlestick series
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#10b981',
          downColor: '#ef4444',
          borderVisible: true,
          borderUpColor: '#10b981',
          borderDownColor: '#ef4444',
          wickUpColor: '#10b981',
          wickDownColor: '#ef4444',
        });

        // Set data
        candlestickSeries.setData(data);

        // Fit content to show all candles properly
        chart.timeScale().fitContent();

        // Store references
        chartRef.current = chart;
        seriesRef.current = candlestickSeries;
        setIsReady(true);
        setError(null);

        // Handle resize
        const handleResize = () => {
          if (chartContainerRef.current && chartRef.current) {
            const newWidth = chartContainerRef.current.clientWidth;
            if (newWidth > 0) {
              chartRef.current.applyOptions({ width: newWidth });
              chartRef.current.timeScale().fitContent();
            }
          }
        };

        window.addEventListener('resize', handleResize);

        return () => {
          window.removeEventListener('resize', handleResize);
          if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
          }
          setIsReady(false);
        };
      } catch (err: any) {
        console.error('Error creating candlestick chart:', err);
        setError(err.message || 'Failed to load chart');
        setIsReady(false);
      }
    };

    checkDimensions();
  }, [height, data]);

  // Update data when it changes
  useEffect(() => {
    if (seriesRef.current && chartRef.current && data && data.length > 0 && isReady) {
      try {
        seriesRef.current.setData(data);
        chartRef.current.timeScale().fitContent();
      } catch (err) {
        console.error('Error updating chart data:', err);
      }
    }
  }, [data, isReady]);

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
        <Typography variant="body2">{error}</Typography>
        <Typography variant="caption" sx={{ color: '#475569' }}>Please refresh the page</Typography>
      </Box>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Box sx={{
        height: `${height}px`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#64748b',
      }}>
        <Typography>Loading chart data...</Typography>
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
        minWidth: '300px',
      }}
    />
  );
}
