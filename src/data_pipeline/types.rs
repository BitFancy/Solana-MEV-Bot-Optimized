// src/data_pipeline/types.rs

use serde::{Serialize, Deserialize};

/// Represents the type of a market event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketEventType {
    Trade,
    OrderBookUpdate,
    // Could add more types like LiquidityChange, NewPair, etc.
}

/// Represents a single raw market event as fetched from an exchange or data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawMarketEvent {
    pub timestamp_ms: u64,
    pub source_exchange: String, // e.g., "Orca", "Raydium", "Birdeye"
    pub pair_address: String,    // Pool address or market ID
    pub event_type: MarketEventType,
    pub price: f64,              // Price of the trade or current price for order book update
    pub volume: f64,             // Volume of the trade or volume at a certain depth for OB update
    pub raw_data: String,        // JSON string for any other specific data from the source
}

impl RawMarketEvent {
    pub fn new(
        timestamp_ms: u64,
        source_exchange: String,
        pair_address: String,
        event_type: MarketEventType,
        price: f64,
        volume: f64,
        raw_data: String,
    ) -> Self {
        Self {
            timestamp_ms,
            source_exchange,
            pair_address,
            event_type,
            price,
            volume,
            raw_data,
        }
    }
}

/// Represents processed market data, often aggregated into candles (OHLCV).
/// Similar to MarketDataPoint in ai/predictor.rs but potentially more comprehensive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessedMarketData {
    pub timestamp_ms: u64,    // Start timestamp of the candle/period
    pub pair_address: String, // Pool address or market ID
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,            // Volume Weighted Average Price for the period
}

impl ProcessedMarketData {
    // Basic constructor for dummy data
    pub fn dummy(timestamp_ms: u64, pair_address: String, price: f64, volume: f64) -> Self {
        Self {
            timestamp_ms,
            pair_address,
            open: price,
            high: price,
            low: price,
            close: price,
            volume,
            vwap: price, // Simplified VWAP for dummy data
        }
    }
}
