import React from 'react'
import { Box } from '@mui/material'
import '../styles/premium.css'

interface TickerItem {
    symbol: string
    price: string
    change: string
}

const defaultTickerItems: TickerItem[] = [
    { symbol: "SPY", price: "478.20", change: "+0.45%" },
    { symbol: "QQQ", price: "412.50", change: "+0.82%" },
    { symbol: "BTC", price: "45,230", change: "+2.1%" },
    { symbol: "ETH", price: "2,405", change: "+1.8%" },
    { symbol: "VIX", price: "13.45", change: "-4.2%" },
    { symbol: "AAPL", price: "185.92", change: "+1.2%" },
    { symbol: "NVDA", price: "495.22", change: "+3.5%" },
    { symbol: "MSFT", price: "378.91", change: "+0.9%" },
    { symbol: "GOOGL", price: "142.15", change: "+0.6%" },
    { symbol: "TSLA", price: "245.80", change: "-1.2%" },
    { symbol: "EUR/USD", price: "1.095", change: "-0.1%" },
    { symbol: "GOLD", price: "2,045", change: "+0.3%" },
]

interface TickerBarProps {
    items?: TickerItem[]
}

export default function TickerBar({ items = defaultTickerItems }: TickerBarProps) {
    // Double the items for seamless infinite scroll
    const tickerItems = [...items, ...items]

    return (
        <Box className="ticker-bar">
            <div className="ticker-wrap">
                <div className="ticker-move">
                    {tickerItems.map((item, index) => (
                        <span key={index} className="ticker-item">
                            <span className="ticker-symbol">{item.symbol}</span>
                            <span className="ticker-price">{item.price}</span>
                            <span className={`ticker-change ${item.change.startsWith('+') ? 'positive' : 'negative'}`}>
                                {item.change}
                            </span>
                        </span>
                    ))}
                </div>
            </div>
        </Box>
    )
}
