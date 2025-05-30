// src/ai/predictor.rs

use std::collections::VecDeque;
use crate::data_pipeline::types::ProcessedMarketData; // Import ProcessedMarketData

// The old MarketDataPoint struct can be removed or commented out.
// #[derive(Debug, Clone, PartialEq)]
// pub struct MarketDataPoint {
//     pub timestamp: u64, // Unix timestamp in seconds
//     pub price: f64,
//     pub volume: f64,
// }
//
// impl MarketDataPoint {
//     pub fn new(timestamp: u64, price: f64, volume: f64) -> Self {
//         MarketDataPoint { timestamp, price, volume }
//     }
// }

const SHORT_TERM_SMA_PERIOD: usize = 5;
const LONG_TERM_SMA_PERIOD: usize = 10; // Should be greater than short term

/// The MarketPredictor struct, responsible for analyzing market data and making predictions.
#[derive(Debug)]
pub struct MarketPredictor {
    historical_data: VecDeque<ProcessedMarketData>, // Using ProcessedMarketData now
    prediction_model: Option<String>,              // Placeholder for a more complex model
    max_historical_points: usize,                  // Max number of data points to store
    current_volatility: f64,                       // Current calculated market volatility
    current_trend: f64,                            // Current identified market trend (-1: down, 0: neutral, 1: up)
    // Store previous SMA values to detect crossovers
    prev_short_sma: Option<f64>,
    prev_long_sma: Option<f64>,
}

impl MarketPredictor {
    /// Creates a new MarketPredictor.
    ///
    /// # Arguments
    ///
    /// * `max_historical_points` - The maximum number of historical data points to store.
    pub fn new(max_historical_points: usize) -> Self {
        MarketPredictor {
            historical_data: VecDeque::with_capacity(max_historical_points),
            prediction_model: None, // No actual model for now
            max_historical_points,
            current_volatility: 0.0,
            current_trend: 0.0,
            prev_short_sma: None,
            prev_long_sma: None,
        }
    }

    /// Helper function to calculate Simple Moving Average (SMA).
    fn calculate_sma(&self, period: usize) -> Option<f64> {
        if self.historical_data.len() < period {
            return None;
        }
        let sum: f64 = self.historical_data.iter().rev().take(period).map(|dp| dp.close).sum();
        Some(sum / period as f64)
    }


    /// Adds a new data point to historical data.
    /// If the number of points exceeds `max_historical_points`, the oldest is removed.
    /// After adding, it recalculates volatility and trend.
    /// Logs the addition of the data point.
    ///
    /// # Arguments
    ///
    /// * `data_point` - The new `ProcessedMarketData` point to add.
    pub fn add_data_point(&mut self, data_point: ProcessedMarketData) {
        if self.historical_data.len() == self.max_historical_points && self.max_historical_points > 0 {
            self.historical_data.pop_front(); // Remove the oldest point
        }
        log::debug!(
            "Adding data point: Timestamp: {}, Pair: {}, Close: {:.4}, Volume: {:.2}",
            data_point.timestamp_ms, data_point.pair_address, data_point.close, data_point.volume
        );
        self.historical_data.push_back(data_point);

        // Update indicators after new data is added
        self.calculate_volatility();
        self.identify_trend(); // This will also update SMAs for prediction
    }

    /// Predicts price movement using SMA crossover logic.
    /// Logs the prediction inputs and outputs.
    ///
    /// # Arguments
    ///
    /// * `_future_horizon_seconds` - How far into the future to predict (currently unused).
    ///
    /// # Returns
    ///
    /// A tuple: `(predicted_price_change_percentage, confidence_score)`.
    pub fn predict_price_movement(&self, _future_horizon_seconds: u64) -> (f64, f64) {
        if self.historical_data.len() < LONG_TERM_SMA_PERIOD {
            log::debug!("Predicting price movement: Not enough historical data for SMA calculation (< {} points), returning (0.0, 0.0).", LONG_TERM_SMA_PERIOD);
            return (0.0, 0.0); // Not enough data for LMA
        }

        let short_sma = self.calculate_sma(SHORT_TERM_SMA_PERIOD);
        let long_sma = self.calculate_sma(LONG_TERM_SMA_PERIOD);

        let mut predicted_change_percentage = 0.0;
        let mut confidence_score = 0.3; // Base confidence

        if let (Some(current_short_sma), Some(current_long_sma), Some(prev_short), Some(prev_long)) =
            (short_sma, long_sma, self.prev_short_sma, self.prev_long_sma) {

            // Bullish Crossover
            if prev_short <= prev_long && current_short_sma > current_long_sma {
                predicted_change_percentage = 0.02; // Predict 2% increase
                confidence_score = 0.75;
                log::debug!("SMA Bullish Crossover detected: prev_short={:.2}, prev_long={:.2}, current_short={:.2}, current_long={:.2}", prev_short, prev_long, current_short_sma, current_long_sma);
            // Bearish Crossover
            } else if prev_short >= prev_long && current_short_sma < current_long_sma {
                predicted_change_percentage = -0.02; // Predict 2% decrease
                confidence_score = 0.75;
                log::debug!("SMA Bearish Crossover detected: prev_short={:.2}, prev_long={:.2}, current_short={:.2}, current_long={:.2}", prev_short, prev_long, current_short_sma, current_long_sma);
            // SMAs Diverging (trend continuation)
            } else if (current_short_sma > current_long_sma && current_short_sma - current_long_sma > prev_short - prev_long) || // Bullish divergence
                      (current_short_sma < current_long_sma && current_long_sma - current_short_sma > prev_long - prev_short)    // Bearish divergence
            {
                confidence_score = 0.60;
                predicted_change_percentage = if current_short_sma > current_long_sma { 0.01 } else { -0.01 }; // Follow current SMA relation
                 log::debug!("SMA Divergence detected: current_short={:.2}, current_long={:.2}", current_short_sma, current_long_sma);
            // SMAs Converging (trend weakening)
            } else if (current_short_sma > current_long_sma && current_short_sma - current_long_sma < prev_short - prev_long) || // Bullish convergence
                      (current_short_sma < current_long_sma && current_long_sma - current_short_sma < prev_long - prev_short)    // Bearish convergence
            {
                confidence_score = 0.40;
                predicted_change_percentage = 0.0; // Neutral as trend might reverse
                log::debug!("SMA Convergence detected: current_short={:.2}, current_long={:.2}", current_short_sma, current_long_sma);
            } else { // No clear signal from crossover, use current trend (as identified by identify_trend)
                predicted_change_percentage = self.current_trend * 0.005; // Smaller prediction based on general trend
                confidence_score = 0.5;
                log::debug!("No SMA crossover/divergence, using general trend: {:.1}", self.current_trend);
            }
        } else { // Not enough data for previous SMAs or current SMAs
            predicted_change_percentage = self.current_trend * 0.005; // Fallback to simpler trend based prediction
            confidence_score = 0.3; // Lower confidence due to lack of SMA confirmation
            log::debug!("Not enough data for full SMA crossover logic, using general trend: {:.1}", self.current_trend);
        }

        // Factor in volatility (lower volatility might increase confidence in stable trends/crossovers)
        confidence_score -= self.current_volatility * 0.5; // Reduce confidence by half of volatility
        confidence_score = confidence_score.max(0.1).min(0.9); // Clamp confidence

        log::info!(
            "Predicting price movement: ShortSMA={:.2?}, LongSMA={:.2?}, Trend={:.2}, Volatility={:.4} -> Predicted Change={:.4}%, Confidence={:.4}",
            short_sma, long_sma, self.current_trend, self.current_volatility, predicted_change_percentage, confidence_score
        );

        (predicted_change_percentage, confidence_score)
    }

    /// Calculates market volatility based on historical price data (closing prices).
    /// Updates `current_volatility`.
    /// Logs the calculated volatility.
    fn calculate_volatility(&mut self) {
        let prices: Vec<f64> = self.historical_data.iter().map(|dp| dp.close).collect(); // Use close price
        if prices.len() < 2 {
            if self.current_volatility != 0.0 { // Log only if it changes
                log::debug!("Calculating volatility: Not enough data (<2 points), setting volatility to 0.0.");
            }
            self.current_volatility = 0.0;
            return;
        }

        let mean_price: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance: f64 = prices.iter().map(|price| (price - mean_price).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        if mean_price > 0.0 {
            self.current_volatility = (std_dev / mean_price).max(0.0); // Normalized volatility
        } else {
            self.current_volatility = 0.0;
        }
        log::debug!("Calculating volatility: Updated to {:.4} based on {} data points.", self.current_volatility, prices.len());
    }

    /// Identifies the market trend based on SMA relationship and historical price data.
    /// Updates `current_trend` and previous SMA values.
    /// Logs the identified trend.
    fn identify_trend(&mut self) {
        let n = self.historical_data.len();
        let old_general_trend = self.current_trend;

        // Update SMAs first
        let current_short_sma = self.calculate_sma(SHORT_TERM_SMA_PERIOD);
        let current_long_sma = self.calculate_sma(LONG_TERM_SMA_PERIOD);

        if let (Some(s_sma), Some(l_sma)) = (current_short_sma, current_long_sma) {
            if s_sma > l_sma * 1.001 { // Short SMA is significantly above Long SMA
                self.current_trend = 1.0; // Uptrend
            } else if s_sma < l_sma * 0.999 { // Short SMA is significantly below Long SMA
                self.current_trend = -1.0; // Downtrend
            } else {
                self.current_trend = 0.0; // Neutral / Indecisive based on SMAs
            }
            log::debug!("Identifying trend (SMA based): ShortSMA={:.2}, LongSMA={:.2} -> Trend={:.1}", s_sma, l_sma, self.current_trend);
        } else { // Fallback to simpler price comparison if SMAs cannot be calculated
            if n < 2 { // Need at least 2 points for simple trend
                self.current_trend = 0.0;
            } else {
                let first_price = self.historical_data.front().unwrap().close;
                let last_price = self.historical_data.back().unwrap().close;
                if last_price > first_price * 1.001 {
                    self.current_trend = 1.0;
                } else if last_price < first_price * 0.999 {
                    self.current_trend = -1.0;
                } else {
                    self.current_trend = 0.0;
                }
            }
            log::debug!("Identifying trend (fallback price comparison): Trend={:.1} based on {} data points.", self.current_trend, n);
        }

        // Update previous SMA values for crossover detection in the next step
        self.prev_short_sma = current_short_sma;
        self.prev_long_sma = current_long_sma;

        if old_general_trend != self.current_trend || (n == LONG_TERM_SMA_PERIOD && (current_short_sma.is_some() && current_long_sma.is_some())) {
             log::info!("General market trend updated to: {:.1}", self.current_trend);
        }
    }

    /// Placeholder for training a prediction model.
    pub fn train_model(&mut self, _training_data: &Vec<ProcessedMarketData>) -> Result<(), String> {
        log::info!("Training prediction model... (Placeholder)");
        // In a real implementation, this would train an ML model or optimize parameters.
        // For SMA, one could optimize periods, but that's out of scope for now.
        self.prediction_model = Some("sma_crossover_v1".to_string()); // Example model name
        Ok(())
    }

    /// Placeholder for loading a prediction model.
    pub fn load_model(&mut self, path: &str) -> Result<(), String> {
        log::info!("Loading prediction model from {}... (Placeholder)", path);
        // In a real implementation, this would deserialize a trained model.
        self.prediction_model = Some(format!("loaded_model_from_{}", path));
        Ok(())
    }

    /// Placeholder for saving a prediction model.
    pub fn save_model(&self, path: &str) -> Result<(), String> {
        log::info!("Saving prediction model to {}... (Placeholder)", path);
        // In a real implementation, this would serialize the trained model.
        if self.prediction_model.is_none() {
            return Err("No model to save.".to_string());
        }
        Ok(())
    }


    /// Returns the current calculated volatility.
    pub fn get_volatility(&self) -> f64 {
        self.current_volatility
    }

    /// Returns the current identified trend.
    pub fn get_trend(&self) -> f64 {
        self.current_trend
    }

    /// Returns the `num_states` most recent market data points.
    /// If `num_states` is larger than available data, returns all available data.
    pub fn get_sliding_window_states(&self, num_states: usize) -> Vec<ProcessedMarketData> { // Changed return type
        let available_data_count = self.historical_data.len();
        let start_index = if num_states >= available_data_count {
            0
        } else {
            available_data_count - num_states
        };
        self.historical_data.iter().skip(start_index).cloned().collect()
    }
}

// Unit tests for MarketPredictor
#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_pipeline::types::ProcessedMarketData; // Ensure this import is correct

    // Helper to create sample ProcessedMarketData
    fn create_sample_processed_data(timestamp_ms: u64, pair_address: &str, close_price: f64, volume: f64) -> ProcessedMarketData {
        ProcessedMarketData {
            timestamp_ms,
            pair_address: pair_address.to_string(),
            open: close_price - 0.5, // Example values
            high: close_price + 0.5,
            low: close_price - 1.0,
            close: close_price,
            volume,
            vwap: close_price, // Simplified
        }
    }

    #[test]
    fn test_predictor_creation_processed_data() {
        let predictor = MarketPredictor::new(100);
        assert_eq!(predictor.max_historical_points, 100);
        assert!(predictor.historical_data.is_empty());
        assert_eq!(predictor.current_volatility, 0.0);
        assert_eq!(predictor.current_trend, 0.0);
        assert!(predictor.prev_short_sma.is_none());
        assert!(predictor.prev_long_sma.is_none());
    }

    #[test]
    fn test_add_data_point_processed_data() {
        let mut predictor = MarketPredictor::new(3);
        let data1 = create_sample_processed_data(1000, "SOL/USDC", 10.0, 100.0);
        let data2 = create_sample_processed_data(2000, "SOL/USDC", 11.0, 110.0);
        let data3 = create_sample_processed_data(3000, "SOL/USDC", 12.0, 120.0);
        predictor.add_data_point(data1.clone());
        predictor.add_data_point(data2.clone());
        predictor.add_data_point(data3.clone());
        assert_eq!(predictor.historical_data.len(), 3);

        let data4 = create_sample_processed_data(4000, "SOL/USDC", 13.0, 130.0);
        predictor.add_data_point(data4.clone()); // This should evict data1
        assert_eq!(predictor.historical_data.len(), 3);
        assert_eq!(predictor.historical_data.front().unwrap().timestamp_ms, 2000); // Oldest is now data2
        assert_eq!(predictor.historical_data.back().unwrap().timestamp_ms, 4000);  // Newest is data4
    }

    #[test]
    fn test_calculate_volatility_processed_data() {
        let mut predictor = MarketPredictor::new(10);
        predictor.add_data_point(create_sample_processed_data(1, "S", 100.0, 1.0));
        predictor.add_data_point(create_sample_processed_data(2, "S", 100.0, 1.0));
        predictor.add_data_point(create_sample_processed_data(3, "S", 100.0, 1.0));
        assert_eq!(predictor.current_volatility, 0.0); // Zero volatility for identical close prices

        let mut predictor_var = MarketPredictor::new(10);
        predictor_var.add_data_point(create_sample_processed_data(1, "S", 100.0, 1.0));
        predictor_var.add_data_point(create_sample_processed_data(2, "S", 102.0, 1.0));
        predictor_var.add_data_point(create_sample_processed_data(3, "S", 104.0, 1.0));
        // Prices: 100, 102, 104. Mean: 102. StdDev: sqrt(8/3) approx 1.633. Vol: 1.633 / 102
        assert!((predictor_var.current_volatility - (8.0f64/3.0).sqrt() / 102.0).abs() < 1e-6);
    }

    #[test]
    fn test_identify_trend_sma_based() {
        let mut predictor = MarketPredictor::new(LONG_TERM_SMA_PERIOD + 5); // Ensure enough space for SMA calculations + history

        // Simulate uptrend by SMA crossover
        // Prices to create LMA < SMA
        let prices_uptrend = [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 100.0, 101.0, 102.0, 103.0, 104.0]; // Last 5 are higher
        for (i, &price) in prices_uptrend.iter().enumerate() {
            predictor.add_data_point(create_sample_processed_data(i as u64 * 1000, "P", price, 100.0));
        }
        // After these points, identify_trend is called by add_data_point.
        // Short SMA (last 5 of 100-104, avg ~102) should be > Long SMA (last 10 of 90-104, avg ~97.5)
        assert_eq!(predictor.get_trend(), 1.0, "Should identify uptrend based on SMA");

        // Simulate downtrend
        predictor.historical_data.clear(); // Reset
        let prices_downtrend = [110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 100.0, 99.0, 98.0, 97.0, 96.0]; // Last 5 are lower
         for (i, &price) in prices_downtrend.iter().enumerate() {
            predictor.add_data_point(create_sample_processed_data(i as u64 * 1000, "P", price, 100.0));
        }
        // Short SMA (last 5 of 96-100, avg ~98) should be < Long SMA (last 10 of 96-110, avg ~103.5)
        assert_eq!(predictor.get_trend(), -1.0, "Should identify downtrend based on SMA");

        // Simulate neutral / flat trend
        predictor.historical_data.clear();
        let prices_neutral = [100.0, 100.1, 100.0, 99.9, 100.0, 100.2, 99.8, 100.0, 100.1, 99.9, 100.0];
        for (i, &price) in prices_neutral.iter().enumerate() {
            predictor.add_data_point(create_sample_processed_data(i as u64 * 1000, "P", price, 100.0));
        }
        // SMAs should be very close
        assert_eq!(predictor.get_trend(), 0.0, "Should identify neutral trend based on SMA");
    }

    #[test]
    fn test_predict_price_movement_sma_crossover() {
        let mut predictor = MarketPredictor::new(LONG_TERM_SMA_PERIOD + 5);

        // Initial flat data
        for i in 0..LONG_TERM_SMA_PERIOD {
            predictor.add_data_point(create_sample_processed_data(i as u64 * 1000, "P", 100.0, 100.0));
        }
        let (change_flat, conf_flat) = predictor.predict_price_movement(60);
        // Before any crossover, prediction might be neutral or based on slight variations
        log::debug!("Flat prediction: change {}, conf {}", change_flat, conf_flat);
        // Assertions depend on exact initial state of SMAs and trend; could be 0 change, low confidence

        // Simulate data for a bullish crossover
        // prev_short_sma and prev_long_sma were set by the flat data.
        // Now add points to make short_sma cross above long_sma
        // Example: LMA around 100. Make last few points higher to pull up SMA.
        predictor.add_data_point(create_sample_processed_data(LONG_TERM_SMA_PERIOD as u64 * 1000, "P", 101.0, 100.0));
        predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 1) as u64 * 1000, "P", 102.0, 100.0));
        predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 2) as u64 * 1000, "P", 103.0, 100.0));
        predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 3) as u64 * 1000, "P", 104.0, 100.0));
        predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 4) as u64 * 1000, "P", 105.0, 100.0));
        // identify_trend is called with each add_data_point, updating prev_short_sma and prev_long_sma

        let (change_bull, conf_bull) = predictor.predict_price_movement(60);
        assert!(change_bull > 0.0, "Bullish crossover should predict positive change. Got {}", change_bull);
        assert!(conf_bull >= 0.70, "Bullish crossover should have high confidence. Got {}", conf_bull); // Check against 0.75 - volatility factor

        // Simulate data for a bearish crossover
        // Reset to a state where short SMA > long SMA, then make it cross below
        predictor.historical_data.clear();
        predictor.prev_short_sma = None; predictor.prev_long_sma = None; // Reset prev SMAs
        for i in 0..LONG_TERM_SMA_PERIOD { // Establish uptrend SMAs
            predictor.add_data_point(create_sample_processed_data(i as u64 * 1000, "P", 110.0 + i as f64 *0.2, 100.0));
        }
         predictor.add_data_point(create_sample_processed_data(LONG_TERM_SMA_PERIOD as u64 * 1000, "P", 100.0, 100.0));
         predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 1) as u64 * 1000, "P", 99.0, 100.0));
         predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 2) as u64 * 1000, "P", 98.0, 100.0));
         predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 3) as u64 * 1000, "P", 97.0, 100.0));
         predictor.add_data_point(create_sample_processed_data((LONG_TERM_SMA_PERIOD + 4) as u64 * 1000, "P", 96.0, 100.0));

        let (change_bear, conf_bear) = predictor.predict_price_movement(60);
        assert!(change_bear < 0.0, "Bearish crossover should predict negative change. Got {}", change_bear);
        assert!(conf_bear >= 0.70, "Bearish crossover should have high confidence. Got {}", conf_bear);
    }

    #[test]
    fn test_get_sliding_window_states_processed_data() {
        let mut predictor = MarketPredictor::new(10);
        let data_points: Vec<ProcessedMarketData> = (0..5)
            .map(|i| create_sample_processed_data(1600000000 + i * 1000, "SOL/USDC", 100.0 + i as f64, 10.0 + i as f64))
            .collect();
        for p in data_points.iter() {
            predictor.add_data_point(p.clone());
        }

        let window1 = predictor.get_sliding_window_states(3);
        assert_eq!(window1.len(), 3);
        assert_eq!(window1[0].timestamp_ms, 1600000000 + 2 * 1000);
    }

    #[test]
    fn test_model_management_placeholders() {
        let mut predictor = MarketPredictor::new(10);
        let empty_data: Vec<ProcessedMarketData> = Vec::new();

        assert!(predictor.train_model(&empty_data).is_ok());
        assert!(predictor.prediction_model.is_some());
        assert_eq!(predictor.prediction_model.as_ref().unwrap(), "sma_crossover_v1");

        let model_path = "test_model.dat";
        assert!(predictor.save_model(model_path).is_ok());

        assert!(predictor.load_model(model_path).is_ok());
        assert!(predictor.prediction_model.as_ref().unwrap().contains(model_path));

        // Test save_model when no model is set (e.g. after fresh init)
        let fresh_predictor = MarketPredictor::new(10);
        assert!(fresh_predictor.save_model(model_path).is_err(), "Saving model should fail if no model is set/trained.");

    }
}
