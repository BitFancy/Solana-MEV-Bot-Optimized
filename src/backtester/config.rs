// src/backtester/config.rs

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIAgentConfig {
    pub risk_tolerance: f64,
    pub min_profit_threshold: f64,
    pub initial_learning_rate: f64,
    pub rl_veto_threshold: f64, // New field for RL veto
}

impl Default for AIAgentConfig {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            min_profit_threshold: 0.001, // Example: 0.1% profit
            initial_learning_rate: 0.1,
            rl_veto_threshold: -0.5, // Default veto threshold for Q-value
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPredictorConfig {
    pub max_historical_points: usize,
    // Add other predictor-specific configs if any, e.g., SMA periods if configurable
}

impl Default for MarketPredictorConfig {
    fn default() -> Self {
        Self {
            max_historical_points: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyOptimizerConfig {
    pub learning_rate_rl: f64,
    pub discount_factor_rl: f64,
    pub exploration_rate_rl: f64,
    // Add other optimizer-specific configs if any
}

impl Default for StrategyOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate_rl: 0.1,
            discount_factor_rl: 0.9,
            exploration_rate_rl: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub start_time_ms: u64,
    pub end_time_ms: u64,
    pub initial_balance: f64,
    pub pair_address: String,
    pub exchange: String,
    pub ai_agent_config: AIAgentConfig,
    pub market_predictor_config: MarketPredictorConfig,
    pub strategy_optimizer_config: StrategyOptimizerConfig,
    pub data_resolution_secs: u32, // Resolution for fetching historical data
}

impl BacktestConfig {
    pub fn new(
        start_time_ms: u64,
        end_time_ms: u64,
        initial_balance: f64,
        pair_address: String,
        exchange: String,
        data_resolution_secs: u32,
        ai_config: Option<AIAgentConfig>,
        predictor_config: Option<MarketPredictorConfig>,
        optimizer_config: Option<StrategyOptimizerConfig>,
    ) -> Self {
        Self {
            start_time_ms,
            end_time_ms,
            initial_balance,
            pair_address,
            exchange,
            data_resolution_secs,
            ai_agent_config: ai_config.unwrap_or_default(),
            market_predictor_config: predictor_config.unwrap_or_default(),
            strategy_optimizer_config: optimizer_config.unwrap_or_default(),
        }
    }
}
