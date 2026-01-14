import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, CandlestickSeries, LineSeries, Time, AreaSeries } from 'lightweight-charts';
import { Box, Typography } from '@mui/material';

interface Prediction {
    period: string;
    direction: 'UP' | 'DOWN';
    confidence: number;
    predictedPrice?: number;
}

interface GhostCandleChartProps {
    predictions: Prediction[];
    currentPrice: number;
    height?: number;
    timeframe: string;
    lastCandleTime?: number;
}

// Helper to get timeframe interval in seconds
const getTimeframeInterval = (timeframe: string): number => {
    const intervals: Record<string, number> = {
        '1M': 60,
        '5M': 300,
        '15M': 900,
        '1H': 3600,
        '4H': 14400,
        '1D': 86400,
    };
    return intervals[timeframe] || 3600;
};

// Generate prediction labels based on timeframe
export const getPredictionLabels = (timeframe: string, count: number = 5): string[] => {
    const labels: string[] = [];
    const interval = timeframe.replace(/\d+/, '');
    const multiplier = parseInt(timeframe.replace(/\D/g, '')) || 1;

    for (let i = 1; i <= count; i++) {
        const value = multiplier * i;
        labels.push(`+${value}${interval}`);
    }
    return labels;
};

// Calculate predicted prices from predictions
export const calculatePredictedPrices = (
    predictions: Prediction[],
    currentPrice: number
): number[] => {
    const prices: number[] = [];
    let price = currentPrice;

    predictions.forEach((pred) => {
        const movementPercent = (pred.confidence / 100) * 0.012;
        const movement = pred.direction === 'UP' ? price * movementPercent : -price * movementPercent;
        price = price + movement;
        prices.push(Number(price.toFixed(2)));
    });

    return prices;
};

export default function GhostCandleChart({
    predictions,
    currentPrice,
    height = 350,
    timeframe,
    lastCandleTime
}: GhostCandleChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;
        if (!predictions || predictions.length === 0) {
            setError('No predictions available');
            return;
        }

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
                const width = container.clientWidth || 400;
                const chartHeight = height;
                const intervalSeconds = getTimeframeInterval(timeframe);

                // Calculate optimal bar spacing for 5 candles
                const numCandles = predictions.length;
                const availableWidth = width - 80; // Reserve space for price scale
                const optimalBarSpacing = Math.max(20, availableWidth / (numCandles * 2));

                // Create chart with better display settings
                const chart = createChart(container, {
                    layout: {
                        background: { color: 'transparent' },
                        textColor: '#94a3b8',
                        fontFamily: 'Inter, system-ui, sans-serif',
                    },
                    grid: {
                        vertLines: { color: 'rgba(139, 92, 246, 0.06)' },
                        horzLines: { color: 'rgba(139, 92, 246, 0.06)' },
                    },
                    width: width,
                    height: chartHeight,
                    timeScale: {
                        timeVisible: true,
                        secondsVisible: false,
                        borderColor: 'rgba(139, 92, 246, 0.12)',
                        barSpacing: optimalBarSpacing,
                        rightOffset: 2,
                        minBarSpacing: 15,
                        fixLeftEdge: true,
                        fixRightEdge: true,
                    },
                    rightPriceScale: {
                        borderColor: 'rgba(139, 92, 246, 0.12)',
                        scaleMargins: {
                            top: 0.1,
                            bottom: 0.1,
                        },
                        autoScale: true,
                    },
                    crosshair: {
                        vertLine: {
                            color: 'rgba(139, 92, 246, 0.5)',
                            labelBackgroundColor: '#8b5cf6',
                        },
                        horzLine: {
                            color: 'rgba(139, 92, 246, 0.5)',
                            labelBackgroundColor: '#8b5cf6',
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

                // Generate ghost candle data with proper OHLC
                const baseTime = lastCandleTime || Math.floor(Date.now() / 1000);
                let price = currentPrice;

                // Data arrays
                const candleData: { time: Time; open: number; high: number; low: number; close: number }[] = [];
                const upperBandData: { time: Time; value: number }[] = [];
                const lowerBandData: { time: Time; value: number }[] = [];

                predictions.forEach((pred, index) => {
                    const time = (baseTime + (index + 1) * intervalSeconds) as Time;

                    // Calculate price movement
                    const movementPercent = (pred.confidence / 100) * 0.015;
                    const movement = pred.direction === 'UP' ? price * movementPercent : -price * movementPercent;
                    const newPrice = price + movement;

                    // Create realistic OHLC candle
                    const open = price;
                    const close = newPrice;
                    const volatilityFactor = 0.008;
                    const high = Math.max(open, close) * (1 + Math.random() * volatilityFactor);
                    const low = Math.min(open, close) * (1 - Math.random() * volatilityFactor);

                    candleData.push({
                        time,
                        open: Number(open.toFixed(2)),
                        high: Number(high.toFixed(2)),
                        low: Number(low.toFixed(2)),
                        close: Number(close.toFixed(2)),
                    });

                    // Confidence bands (wider for lower confidence)
                    const confidenceWidth = ((100 - pred.confidence) / 100) * 0.02;
                    upperBandData.push({ time, value: Number((high * (1 + confidenceWidth)).toFixed(2)) });
                    lowerBandData.push({ time, value: Number((low * (1 - confidenceWidth)).toFixed(2)) });

                    price = newPrice;
                });

                // Add area series for glow effect (subtle background)
                const areaSeriesData = candleData.map((c) => ({
                    time: c.time,
                    value: (c.high + c.low) / 2
                }));

                const areaSeries = chart.addSeries(AreaSeries, {
                    topColor: 'rgba(139, 92, 246, 0.2)',
                    bottomColor: 'rgba(139, 92, 246, 0.02)',
                    lineColor: 'rgba(139, 92, 246, 0.4)',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                // Add ghost candlesticks - more visible but still "ghost" style
                const candlestickSeries = chart.addSeries(CandlestickSeries, {
                    upColor: 'rgba(16, 185, 129, 0.8)',
                    downColor: 'rgba(239, 68, 68, 0.8)',
                    borderVisible: true,
                    borderUpColor: '#10b981',
                    borderDownColor: '#ef4444',
                    wickUpColor: 'rgba(16, 185, 129, 0.7)',
                    wickDownColor: 'rgba(239, 68, 68, 0.7)',
                });

                // Add confidence bands (dashed lines)
                const upperLine = chart.addSeries(LineSeries, {
                    color: 'rgba(139, 92, 246, 0.4)',
                    lineWidth: 1,
                    lineStyle: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                const lowerLine = chart.addSeries(LineSeries, {
                    color: 'rgba(139, 92, 246, 0.4)',
                    lineWidth: 1,
                    lineStyle: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });

                // Set all data
                areaSeries.setData(areaSeriesData);
                candlestickSeries.setData(candleData);
                upperLine.setData(upperBandData);
                lowerLine.setData(lowerBandData);

                // Fit content to fill the chart properly
                chart.timeScale().fitContent();

                // Store reference
                chartRef.current = chart;
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
                };
            } catch (err: any) {
                console.error('Error creating ghost candle chart:', err);
                setError(err.message || 'Failed to load chart');
            }
        };

        checkDimensions();
    }, [predictions, currentPrice, height, timeframe, lastCandleTime]);

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
                background: 'rgba(0,0,0,0.15)',
                borderRadius: 2,
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
                background: 'rgba(0,0,0,0.15)',
                borderRadius: 2,
            }}>
                <Typography sx={{ fontSize: '0.85rem' }}>Loading predictions...</Typography>
            </Box>
        );
    }

    return (
        <Box
            ref={chartContainerRef}
            sx={{
                width: '100%',
                height: `${height}px`,
                position: 'relative',
                minWidth: '200px',
                borderRadius: 2,
                overflow: 'hidden',
            }}
        />
    );
}
