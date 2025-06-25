//! # Enhanced Selling Strategy for Solana Trading Bot
//!
//! This module implements a sophisticated selling strategy for tokens on Solana DEXes,
//! specifically targeting PumpFun and PumpSwap protocols. The strategy combines multiple
//! approaches to determine optimal exit points:
//!
//! ## Core Components
//!
//! 1. **Price-Based Analysis**
//!    - Take profit targets
//!    - Stop loss protection
//!    - Trailing stops that activate after reaching profit thresholds
//!    - Technical indicators (EMA crossovers)
//!
//! 2. **Liquidity Monitoring**
//!    - Minimum absolute liquidity thresholds
//!    - Relative liquidity drop detection
//!    - Price impact assessment
//!
//! 3. **Volume Analysis**
//!    - Volume spike detection (potential exit signals)
//!    - Volume drop detection (liquidity drying up)
//!    - Moving averages of volume to identify trends
//!
//! 4. **Time-Based Exits**
//!    - Maximum hold time to prevent bag holding
//!    - Minimum hold time for profitable positions
//!
//! 5. **Manipulation Detection**
//!    - Wash trading detection
//!    - Creator selling detection
//!    - Large holder movement analysis
//!
//! 6. **Strategy Adaptation**
//!    - Market condition assessment (Bullish, Bearish, Volatile, Stable)
//!    - Dynamic adjustment of parameters based on market conditions
//!
//! 7. **Execution Management**
//!    - Progressive selling in chunks
//!    - Dynamic slippage calculation
//!    - Protocol selection between PumpFun and PumpSwap
//!
//! 8. **Backtesting Framework**
//!    - Historical trade analysis
//!    - Performance reporting
//!    - Strategy optimization
//!
//! The strategy maintains comprehensive metrics on each token position and
//! employs a multi-factor decision framework to determine when to exit positions.

use crate::common::config::import_env_var;
use crate::engine::monitor::PoolInfo;
use solana_sdk::signature::Signer;
use std::collections::{HashMap, HashSet, VecDeque};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::{anyhow, Result};
use anchor_client::solana_sdk::{hash::Hash, instruction::Instruction, pubkey::Pubkey, signature::{Keypair, Signature}};
use colored::Colorize;
use tokio::time::sleep;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::common::{
    config::{AppState, SwapConfig},
    logger::Logger,
};
use crate::dex::pump_fun::Pump;
use crate::dex::pump_swap::PumpSwap;
use crate::engine::swap::{SwapDirection, SwapInType, SwapProtocol};
use crate::engine::transaction_parser::{ParsedData, DexType, TradeInfoFromToken};

// Global state for token metrics
lazy_static! {
    static ref TOKEN_METRICS: Arc<Mutex<HashMap<String, TokenMetrics>>> = Arc::new(Mutex::new(HashMap::new()));
    static ref TOKEN_TRACKING: Arc<Mutex<HashMap<String, TokenTrackingInfo>>> = Arc::new(Mutex::new(HashMap::new()));
    static ref HISTORICAL_TRADES: Arc<Mutex<VecDeque<TradeExecutionRecord>>> = Arc::new(Mutex::new(VecDeque::with_capacity(100)));
}

/// Token metrics for selling strategy
#[derive(Clone, Debug)]
pub struct TokenMetrics {
    pub entry_price: f64,
    pub highest_price: f64,
    pub lowest_price: f64,
    pub current_price: f64,
    pub volume_24h: f64,
    pub market_cap: f64,
    pub time_held: u64,
    pub last_update: Instant,
    pub buy_timestamp: u64,
    pub amount_held: f64,
    pub cost_basis: f64,
    pub price_history: VecDeque<f64>,     // Rolling window of prices
    pub volume_history: VecDeque<f64>,    // Rolling window of volumes
    pub liquidity_at_entry: f64,
}

/// Token tracking info for progressive selling
pub struct TokenTrackingInfo {
    pub top_pnl: f64,
    pub last_sell_time: Instant,
    pub completed_intervals: HashSet<String>,
    pub sell_attempts: usize,
    pub sell_success: usize,
}

/// Record of executed trades for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeExecutionRecord {
    pub mint: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub reason: String,
    pub timestamp: u64,
    pub amount_sold: f64,
    pub protocol: String,
}

/// Market condition enum for dynamic strategy adjustment
#[derive(Debug, Clone)]
pub enum MarketCondition {
    Bullish,
    Bearish,
    Volatile,
    Stable,
}

/// Configuration for profit taking strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitTakingConfig {
    pub target_percentage: f64,           // 1.0 = 100%
    pub scale_out_percentages: Vec<f64>,  // [0.5, 0.3, 0.2] for 50%, 30%, 20%
}

impl Default for ProfitTakingConfig {
    fn default() -> Self {
        Self {
            target_percentage: 1.0,       // 100% profit target
            scale_out_percentages: vec![0.5, 0.3, 0.2], // 50%, 30%, 20%
        }
    }
}

/// Configuration for trailing stop strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopConfig {
    pub activation_percentage: f64,   // 0.2 = 20% from peak
    pub trail_percentage: f64,        // 0.05 = 5% trailing
}

impl Default for TrailingStopConfig {
    fn default() -> Self {
        Self {
            activation_percentage: 0.2,   // 20% activation threshold
            trail_percentage: 0.05,       // 5% trail
        }
    }
}

/// Configuration for liquidity monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMonitorConfig {
    pub min_absolute_liquidity: f64,  // Minimum SOL liquidity to hold
    pub max_acceptable_drop: f64,     // 0.5 = 50% drop from entry
}

impl Default for LiquidityMonitorConfig {
    fn default() -> Self {
        Self {
            min_absolute_liquidity: 1.0,  // 1 SOL minimum liquidity
            max_acceptable_drop: 0.5,     // 50% drop from entry
        }
    }
}

/// Configuration for volume analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnalysisConfig {
    pub lookback_period: usize,       // Number of trades to look back
    pub spike_threshold: f64,         // 3.0 = 3x average volume
    pub drop_threshold: f64,          // 0.3 = 30% of average volume
}

impl Default for VolumeAnalysisConfig {
    fn default() -> Self {
        Self {
            lookback_period: 20,      // 20 trades lookback
            spike_threshold: 3.0,     // 3x average volume for spike
            drop_threshold: 0.3,      // 30% of average volume for drop
        }
    }
}

/// Configuration for time-based exits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeExitConfig {
    pub max_hold_time_secs: u64,      // Maximum time to hold position
    pub min_profit_time_secs: u64,    // Minimum time to hold profitable trades
}

impl Default for TimeExitConfig {
    fn default() -> Self {
        Self {
            max_hold_time_secs: 3600,     // 1 hour max hold time
            min_profit_time_secs: 120,    // 2 minutes min hold for profitable trades
        }
    }
}

/// Configuration for selling strategy
#[derive(Clone, Debug)]
pub struct SellingConfig {
    pub take_profit: f64,       // Percentage (e.g., 2.0 for 2%)
    pub stop_loss: f64,         // Percentage (e.g., -5.0 for -5%)
    pub max_hold_time: u64,     // Seconds
    pub retracement_threshold: f64, // Percentage drop from highest price
    pub min_liquidity: f64,     // Minimum SOL in pool
    pub copy_trade_wallets: Vec<Pubkey>,
    pub progressive_sell_chunks: usize, // Number of chunks for progressive selling
    pub progressive_sell_interval: u64, // Seconds between sells
    
    // Enhanced selling strategy configurations
    pub profit_taking: ProfitTakingConfig,
    pub trailing_stop: TrailingStopConfig,
    pub liquidity_monitor: LiquidityMonitorConfig,
    pub volume_analysis: VolumeAnalysisConfig,
    pub time_based: TimeExitConfig,
    // pub is_progressive_sell: bool,
}

impl Default for SellingConfig {
    fn default() -> Self {
        Self {
            take_profit: 2.0,                // 2% profit target
            stop_loss: -5.0,                 // 5% stop loss
            max_hold_time: 3600,             // 1 hour max hold time
            retracement_threshold: 5.0,      // 5% retracement threshold
            min_liquidity: 1.0,              // 1 SOL minimum liquidity
            copy_trade_wallets: vec![],      // No specific wallets to copy
            progressive_sell_chunks: 3,      // Sell in 3 chunks
            progressive_sell_interval: 120,  // 2 minutes between sells
            
            // Enhanced selling strategy configurations
            profit_taking: ProfitTakingConfig::default(),
            trailing_stop: TrailingStopConfig::default(),
            liquidity_monitor: LiquidityMonitorConfig::default(),
            volume_analysis: VolumeAnalysisConfig::default(),
            time_based: TimeExitConfig::default(),
            // is_progressive_sell: false,
        }
    }
}

/// Status of a token being managed
#[derive(Debug, Clone, PartialEq)]
pub enum TokenStatus {
    Active,           // Token is actively being managed
    PendingSell,      // Token is in process of being sold
    Sold,             // Token has been completely sold
    Failed,           // Token transaction failed
}

/// Token Manager to track and manage multiple tokens
#[derive(Clone)]
pub struct TokenManager {
    logger: Logger,
}

impl TokenManager {
    pub fn new() -> Self {
        Self {
            logger: Logger::new("[TOKEN-MANAGER] => ".cyan().to_string()),
        }
    }

    /// Get a list of all active token mints
    pub fn get_active_tokens(&self) -> Vec<String> {
        let metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        metrics_map.keys().cloned().collect()
    }

    /// Check if a token exists in the metrics
    pub fn token_exists(&self, token_mint: &str) -> bool {
        let metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        metrics_map.contains_key(token_mint)
    }

    /// Get metrics for a specific token, if it exists
    pub fn get_token_metrics(&self, token_mint: &str) -> Option<TokenMetrics> {
        let metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        metrics_map.get(token_mint).cloned()
    }

    /// Add or update a token in the metrics map
    pub fn update_token(&self, token_mint: &str, metrics: TokenMetrics) -> Result<()> {
        let mut metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        metrics_map.insert(token_mint.to_string(), metrics);
        
        self.logger.log(format!("Updated metrics for token: {}", token_mint));
        Ok(())
    }

    /// Remove a token from tracking
    pub fn remove_token(&self, token_mint: &str) -> Result<()> {
        let mut metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        
        if metrics_map.remove(token_mint).is_some() {
            // Also remove from tracking
            let mut tracking = TOKEN_TRACKING.lock().expect("Failed to lock token tracking");
            tracking.remove(token_mint);
            
            self.logger.log(format!("Removed token from tracking: {}", token_mint));
        } else {
            self.logger.log(format!("Token not found for removal: {}", token_mint));
        }
        
        Ok(())
    }
    
    /// Log current token portfolio status
    pub fn log_token_portfolio(&self) {
        let metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        
        if metrics_map.is_empty() {
            self.logger.log("No tokens currently in portfolio".yellow().to_string());
            return;
        }
        
        self.logger.log(format!("Current portfolio contains {} tokens:", metrics_map.len()).green().to_string());
        
        for (mint, metrics) in metrics_map.iter() {
            let current_pnl = if metrics.entry_price > 0.0 {
                ((metrics.current_price - metrics.entry_price) / metrics.entry_price) * 100.0
            } else {
                0.0
            };
            
            let pnl_color = if current_pnl >= 0.0 { "green" } else { "red" };
            
            self.logger.log(format!(
                "Token: {} - Amount: {:.2}, Entry: {:.8}, Current: {:.8}, PNL: {}",
                mint,
                metrics.amount_held,
                metrics.entry_price,
                metrics.current_price,
                format!("{:.2}%", current_pnl).color(pnl_color).to_string()
            ));
        }
    }
    
    /// Monitor all tokens and identify which ones need action
    pub async fn monitor_all_tokens(&self, engine: &SellingEngine) -> Result<()> {
        let tokens = self.get_active_tokens();
        
        if tokens.is_empty() {
            self.logger.log("No tokens to monitor".yellow().to_string());
            return Ok(());
        }
        
        self.logger.log(format!("Monitoring {} tokens", tokens.len()).blue().to_string());
        
        for token_mint in tokens {
            match engine.evaluate_sell_conditions(&token_mint).await {
                Ok(should_sell) => {
                    if should_sell {
                        self.logger.log(format!("Token {} should be sold", token_mint).yellow().to_string());
                        
                        // Execute sell in a separate task to not block monitoring of other tokens
                        let engine_clone = engine.clone();
                        let token_mint_clone = token_mint.clone();
                        tokio::spawn(async move {
                            if let Err(e) = engine_clone.execute_progressive_sell(&token_mint_clone).await {
                                let logger = Logger::new("[TOKEN-MANAGER-SELL] => ".red().to_string());
                                logger.log(format!("Failed to sell token {}: {}", token_mint_clone, e));
                            }
                        });
                    }
                },
                Err(e) => {
                    self.logger.log(format!("Error evaluating sell conditions for {}: {}", token_mint, e).red().to_string());
                }
            }
        }
        
        Ok(())
    }
}

/// Engine for executing selling strategies
#[derive(Clone)]
pub struct SellingEngine {
    app_state: Arc<AppState>,
    swap_config: Arc<SwapConfig>,
    config: SellingConfig,
    logger: Logger,
    token_manager: TokenManager,
    is_progressive_sell: bool,
}

impl SellingEngine {
    pub fn new(
        app_state: Arc<AppState>,
        swap_config: Arc<SwapConfig>,
        config: SellingConfig,
        is_progressive_sell: bool,
    ) -> Self {
        Self {
            app_state,
            swap_config,
            config,
            logger: Logger::new("[SELLING-STRATEGY] => ".yellow().to_string()),
            token_manager: TokenManager::new(),
            is_progressive_sell: is_progressive_sell,
        }
    }
    
    /// Get the token manager
    pub fn token_manager(&self) -> &TokenManager {
        &self.token_manager
    }
    
    /// Get a list of all tokens being managed
    pub fn get_active_tokens(&self) -> Vec<String> {
        self.token_manager.get_active_tokens()
    }
    
    /// Log the current token portfolio
    pub fn log_token_portfolio(&self) {
        self.token_manager.log_token_portfolio();
    }
    
    /// Monitor all tokens and sell if needed
    pub async fn monitor_all_tokens(&self) -> Result<()> {
        self.token_manager.monitor_all_tokens(self).await
    }
    
    /// Update metrics for a token based on parsed transaction data
    pub fn update_metrics(&self, token_mint: &str, parsed_data: &ParsedData) -> Result<()> {
        let logger = Logger::new("[SELLING-STRATEGY] => ".magenta().to_string());
        
        // Extract data
        let sol_change = parsed_data.sol_change;
        let token_change = parsed_data.token_change;
        let is_buy = parsed_data.is_buy;
        let timestamp = parsed_data.timestamp.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });
        
        // Calculate price if possible
        let price = if token_change != 0.0 && sol_change != 0.0 {
            (sol_change / token_change).abs()
        } else {
            0.0
        };
        
        if price <= 0.0 {
            logger.log(format!("Invalid price calculated: {}", price));
            return Err(anyhow::anyhow!("Invalid price calculation"));
        }
        
        // Update token metrics
        let mut metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        let metrics = metrics_map.entry(token_mint.to_string())
            .or_insert_with(|| TokenMetrics {
                entry_price: 0.0,
                highest_price: 0.0,
                lowest_price: 0.0, // Add the lowest_price field
                current_price: 0.0,
                volume_24h: 0.0,
                market_cap: 0.0,
                time_held: 0,
                last_update: Instant::now(),
                buy_timestamp: timestamp,
                amount_held: 0.0,
                cost_basis: 0.0,
                price_history: VecDeque::new(),
                volume_history: VecDeque::new(),
                liquidity_at_entry: 0.0,
            });
        
        // Update metrics based on transaction type
        if is_buy {
            // For buy transactions, update entry price if not set
            if metrics.entry_price == 0.0 {
                metrics.entry_price = price;
                metrics.buy_timestamp = timestamp;
                metrics.last_update = Instant::now();
                
                logger.log(format!("Initialized entry price for {}: {}", token_mint, price));
            }
            
            // Update amount held
            if token_change > 0.0 {
                metrics.amount_held += token_change;
                metrics.cost_basis += sol_change.abs();
                
                logger.log(format!("Updated token balance: {}, cost basis: {}", 
                          metrics.amount_held, metrics.cost_basis));
            }
        } else {
            // For sell transactions, update amount held
            if token_change < 0.0 {
                // Reduce amount held, but don't go below zero
                metrics.amount_held = (metrics.amount_held - token_change.abs()).max(0.0);
                
                // Reduce cost basis proportionally
                if metrics.amount_held > 0.0 && metrics.cost_basis > 0.0 {
                    let sell_percentage = token_change.abs() / (metrics.amount_held + token_change.abs());
                    metrics.cost_basis *= 1.0 - sell_percentage;
                } else {
                    metrics.cost_basis = 0.0;
                }
                
                logger.log(format!("Updated token balance after sell: {}, cost basis: {}", 
                          metrics.amount_held, metrics.cost_basis));
            }
        }
        
        // Always update current price
        metrics.current_price = price;
        
        // Update highest price if applicable
        if price > metrics.highest_price {
            metrics.highest_price = price;
        }
        
        // Update lowest price if applicable (initialize or update if lower)
        if metrics.lowest_price == 0.0 || price < metrics.lowest_price {
            metrics.lowest_price = price;
        }
        
        // Update price history
        metrics.price_history.push_back(price);
        if metrics.price_history.len() > 20 {  // Keep last 20 prices
            metrics.price_history.pop_front();
        }
        
        // Log current metrics
        let pnl = if metrics.entry_price > 0.0 {
            ((price - metrics.entry_price) / metrics.entry_price) * 100.0
        } else {
            0.0
        };
        
        logger.log(format!(
            "Token metrics for {}: Price: {}, Entry: {}, Highest: {}, Lowest: {}, PNL: {:.2}%",
            token_mint, price, metrics.entry_price, metrics.highest_price, metrics.lowest_price, pnl
        ));
        
        Ok(())
    }
    
    /// Record a buy transaction for a token with enhanced metrics tracking
    pub fn record_buy(&self, token_mint: &str, amount: f64, cost: f64) -> Result<()> {
        let logger = Logger::new("[SELLING-STRATEGY-BUY] => ".green().to_string());
        logger.log(format!("Recording buy: {} tokens for {} SOL", amount, cost));
        
        // Get the current timestamp
        let current_timestamp = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(n) => n.as_secs(),
            Err(_) => 0,
        };
        
        // Get liquidity information asynchronously
        let liquidity = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let protocol = match self.determine_best_protocol_for_token(token_mint).await {
                    Ok(p) => p,
                    Err(_) => return 0.0,
                };
                
                match self.get_pool_liquidity(token_mint).await {
                    Ok(liq) => liq,
                    Err(_) => 0.0,
                }
            })
        });
        
        // Get or create metrics for this token
        let mut metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
        let metrics = metrics_map.entry(token_mint.to_string())
            .or_insert_with(|| TokenMetrics {
                entry_price: 0.0,
                highest_price: 0.0,
                lowest_price: 0.0,
                current_price: 0.0,
                volume_24h: 0.0,
                market_cap: 0.0,
                time_held: 0,
                last_update: Instant::now(),
                buy_timestamp: 0,
                amount_held: 0.0,
                cost_basis: 0.0,
                price_history: VecDeque::new(),
                volume_history: VecDeque::new(),
                liquidity_at_entry: liquidity,
            });
        
        // Calculate entry price
        if amount > 0.0 {
            let entry_price = cost / amount;
            
            // Update metrics
            metrics.entry_price = entry_price;
            metrics.current_price = entry_price;
            metrics.highest_price = entry_price; // Reset highest price on new buy
            metrics.lowest_price = entry_price;  // Reset lowest price on new buy
            metrics.buy_timestamp = current_timestamp;
            metrics.amount_held = amount;
            metrics.cost_basis = cost;
            metrics.time_held = 0; // Reset time held
            metrics.liquidity_at_entry = liquidity;
            
            // Initialize price history
            metrics.price_history.clear();
            metrics.price_history.push_back(entry_price);
            
            // Initialize volume history
            metrics.volume_history.clear();
            
            // Log the update
            logger.log(format!(
                "Recorded buy for {}: {} tokens at {} SOL each, total cost: {} SOL, liquidity: {} SOL",
                token_mint, amount, entry_price, cost, liquidity
            ));
            
            // Initialize token tracking
            let mut tracking = TOKEN_TRACKING.lock().expect("Failed to lock token tracking");
            tracking.entry(token_mint.to_string()).or_insert(TokenTrackingInfo {
                top_pnl: 0.0,
                last_sell_time: Instant::now(),
                completed_intervals: HashSet::new(),
                sell_attempts: 0,
                sell_success: 0,
            });
        } else {
            logger.log(format!("Invalid buy amount for {}: {}", token_mint, amount).red().to_string());
        }
        
        Ok(())
    }
    
    /// Evaluate whether we should sell a token based on various conditions
    /// 
    /// This method combines all selling conditions from the enhanced decision framework
    /// into a single evaluation, providing a comprehensive analysis of when to exit a position.
    pub async fn evaluate_sell_conditions(&self, token_mint: &str) -> Result<bool> {
        // Get metrics for the token
        let metrics = {
            let metrics_map = TOKEN_METRICS.lock().expect("Failed to lock metrics for evaluation");
            match metrics_map.get(token_mint) {
                Some(m) => m.clone(), // Clone the metrics to avoid holding the lock
                None => return Ok(false), // No metrics, so nothing to sell
            }
        };
        
        // Remaining code with metrics available outside the lock
        // Use conditions to determine if we should sell
        
        // Calculate time held
        let time_held = metrics.last_update.elapsed().as_secs();
        
        // Check if we've exceeded the max hold time
        if time_held > self.config.max_hold_time {
            self.logger.log(format!("Selling due to max hold time exceeded: {} > {}", 
                             time_held, self.config.max_hold_time).yellow().to_string());
            return Ok(true);
        }
        
        // Calculate current price vs highest price
        let current_price = match self.get_current_price(token_mint).await {
            Ok(price) => price,
            Err(_) => metrics.current_price, // Fall back to last known price
        };
        
        // Calculate percentage change from highest price
        let retracement = if metrics.highest_price > 0.0 {
            (metrics.highest_price - current_price) / metrics.highest_price * 100.0
        } else {
            0.0
        };
        
        // Calculate percentage gain from entry
        let gain = if metrics.entry_price > 0.0 {
            (current_price - metrics.entry_price) / metrics.entry_price * 100.0
        } else {
            0.0
        };
        
        // Log metrics
        self.logger.log(format!(
            "Token: {} - Current: {:.8}, Entry: {:.8}, Highest: {:.8}, Gain: {:.2}%, Retracement: {:.2}%",
            token_mint, current_price, metrics.entry_price, metrics.highest_price, gain, retracement
        ).blue().to_string());
        
        // Check if we've reached take profit
        if gain >= self.config.take_profit {
            self.logger.log(format!("Selling due to take profit reached: {:.2}% >= {:.2}%", 
                             gain, self.config.take_profit).green().to_string());
            return Ok(true);
        }
        
        // Check if we've hit stop loss
        if gain <= self.config.stop_loss {
            self.logger.log(format!("Selling due to stop loss triggered: {:.2}% <= {:.2}%", 
                             gain, self.config.stop_loss).red().to_string());
            return Ok(true);
        }
        
        // Check if price has retraced too much from highest
        if retracement >= self.config.retracement_threshold && gain > 0.0 {
            self.logger.log(format!(
                "Selling due to excessive retracement: {:.2}% >= {:.2}% (still in profit: {:.2}%)",
                retracement, self.config.retracement_threshold, gain
            ).yellow().to_string());
            return Ok(true);
        }
        
        // Check if liquidity is too low
        let liquidity = match self.get_pool_liquidity(token_mint).await {
            Ok(liq) => liq,
            Err(_) => 0.0, // Conservative approach: if we can't check liquidity, assume it's low
        };
        
        if liquidity < self.config.min_liquidity {
            self.logger.log(format!("Selling due to low liquidity: {:.2} < {:.2}", 
                            liquidity, self.config.min_liquidity).red().to_string());
            return Ok(true);
        }
        
        // Check if copy targets are selling
        match self.is_copy_target_selling(token_mint).await {
            Ok(true) => {
                self.logger.log("Selling because copy targets are selling".yellow().to_string());
                return Ok(true);
            },
            _ => {}, // Ignore errors or false result
        }
        
        // If we've reached here, no sell conditions met
        Ok(false)
    }
    
    /// Check if any copy targets are selling this token
    async fn is_copy_target_selling(&self, _token_mint: &str) -> Result<bool> {
        // This would check if any wallets we're copying are selling this token
        // For now, we'll just return false as a placeholder
        Ok(false)
    }
    
    /// Get the current pool liquidity for a token
    async fn get_pool_liquidity(&self, token_mint: &str) -> Result<f64> {
        // Try to get liquidity from PumpSwap first
        let pump_swap = PumpSwap::new(
            self.app_state.wallet.clone(),
            Some(self.app_state.rpc_client.clone()),
            Some(self.app_state.rpc_nonblocking_client.clone()),
        );
        
        match pump_swap.get_token_price(token_mint).await {
            Ok(_) => {
                // If we got a price, we can assume the pool exists
                // In a real implementation, you'd get the actual SOL reserves here
                return Ok(50.0); // Placeholder value
            },
            Err(_) => {
                // Try PumpFun instead
                let pump_fun = Pump::new(
                    self.app_state.rpc_nonblocking_client.clone(),
                    self.app_state.rpc_client.clone(),
                    self.app_state.wallet.clone(),
                );
                
                match pump_fun.get_token_price(token_mint).await {
                    Ok(_) => Ok(30.0), // Placeholder value
                    Err(e) => Err(anyhow!("Failed to get liquidity: {}", e)),
                }
            }
        }
    }
    
    /// Get the current price of a token
    async fn get_current_price(&self, token_mint: &str) -> Result<f64> {
        // Try to get price from PumpSwap first
        let pump_swap = PumpSwap::new(
            self.app_state.wallet.clone(),
            Some(self.app_state.rpc_client.clone()),
            Some(self.app_state.rpc_nonblocking_client.clone()),
        );
        
        match pump_swap.get_token_price(token_mint).await {
            Ok(price) => Ok(price),
            Err(_) => {
                // Try PumpFun instead
                let pump_fun = Pump::new(
                    self.app_state.rpc_nonblocking_client.clone(),
                    self.app_state.rpc_client.clone(),
                    self.app_state.wallet.clone(),
                );
                
                match pump_fun.get_token_price(token_mint).await {
                    Ok(price) => Ok(price),
                    Err(e) => Err(anyhow!("Failed to get price: {}", e)),
                }
            }
        }
    }
    
    /// Calculate dynamic slippage based on liquidity
    fn calculate_dynamic_slippage(&self, token_mint: &str, sell_amount: f64) -> Result<u64> {
        // Get the data we need in a scoped block to release the lock
        let token_value = {
            let metrics_map = TOKEN_METRICS.lock().unwrap();
            
            // Check if we have metrics for this token
            let metrics = match metrics_map.get(token_mint) {
                Some(m) => m,
                None => return Err(anyhow!("No metrics found for token {}", token_mint)),
            };
            
            // Calculate token value
            sell_amount * metrics.current_price
        }; // Lock is released here
        
        // More valuable tokens need higher slippage to ensure execution
        // Return basis points (100 = 1%)
        let slippage_bps = if token_value > 10.0 {
            300 // 3% for high value tokens (300 basis points)
        } else if token_value > 1.0 {
            200 // 2% for medium value tokens (200 basis points)
        } else {
            100 // 1% for low value tokens (100 basis points)
        };
        
        Ok(slippage_bps)
    }
    
    /// Calculate the optimal amount to sell based on metrics
    fn calculate_optimal_sell_amount(&self, token_mint: &str) -> Result<f64> {
        // Get the data we need in a scoped block to release the lock
        let (amount_held, pnl_percentage) = {
            let metrics_map = TOKEN_METRICS.lock().unwrap();
            
            // Check if we have metrics for this token
            let metrics = match metrics_map.get(token_mint) {
                Some(m) => m,
                None => return Err(anyhow!("No metrics found for token {}", token_mint)),
            };
            
            // Calculate PNL
            let pnl = if metrics.entry_price > 0.0 {
                ((metrics.current_price - metrics.entry_price) / metrics.entry_price) * 100.0
            } else {
                0.0
            };
            
            (metrics.amount_held, pnl)
        }; // Lock is released here
        
        // Calculate sell percentage based on PNL
        let sell_percentage = if pnl_percentage >= 200.0 {
            1.0 // Sell 100% if 200%+ profit
        } else if pnl_percentage >= 100.0 {
            0.8 // Sell 80% if 100%+ profit
        } else if pnl_percentage >= 50.0 {
            0.6 // Sell 60% if 50%+ profit
        } else if pnl_percentage >= 20.0 {
            0.5 // Sell 50% if 20%+ profit
        } else if pnl_percentage > 0.0 {
            0.4 // Sell 40% if any profit
        } else {
            0.9 // Sell 90% if at a loss
        };
        
        // Calculate amount to sell
        let amount_to_sell = amount_held * sell_percentage;
        
        Ok(amount_to_sell)
    }
    
    /// Execute a progressive sell strategy
    pub async fn progressive_sell(&self, token_mint: &str, parsed_data: &TradeInfoFromToken, protocol: SwapProtocol) -> Result<()> {
        self.logger.log(format!("Starting progressive sell for {}", token_mint).yellow().to_string());
        
        // Get optimal sell amount
        let total_amount = self.calculate_optimal_sell_amount(token_mint)?;
        
        // Calculate chunk size
        let chunk_size = total_amount / self.config.progressive_sell_chunks as f64;
        
        // Calculate dynamic slippage
        let slippage = self.calculate_dynamic_slippage(token_mint, chunk_size)?;
        
        self.logger.log(format!(
            "Progressive sell plan: total={:.2}, chunks={}, chunk_size={:.2}, slippage={:.2}%",
            total_amount, self.config.progressive_sell_chunks, chunk_size, (slippage as f64) / 100.0
        ));
        
        // Get recent blockhash for the trade info
        let recent_blockhash = match self.app_state.rpc_nonblocking_client.get_latest_blockhash().await {
            Ok(hash) => hash,
            Err(e) => return Err(anyhow!("Failed to get blockhash: {}", e)),
        };
        
        // Create a TradeInfoFromToken for the sell operation with real data from parsed_data
        let base_trade_info = TradeInfoFromToken {
            signature: "progressive_sell".to_string(),
            mint: token_mint.to_string(),
            dex_type: parsed_data.dex_type.clone(),
            is_buy: false,
            slot: parsed_data.slot,
            recent_blockhash,
            target: parsed_data.target.clone(),
            user: parsed_data.user.clone(),
            timestamp: match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(n) => n.as_secs(),
                Err(_) => 0,
            },
            token_amount_f64: total_amount,
            
            // Copy real values from parsed_data
            sol_amount: parsed_data.sol_amount,
            token_amount: parsed_data.token_amount,
            virtual_sol_reserves: parsed_data.virtual_sol_reserves,
            virtual_token_reserves: parsed_data.virtual_token_reserves,
            real_sol_reserves: parsed_data.real_sol_reserves,
            real_token_reserves: parsed_data.real_token_reserves,
            bonding_curve: parsed_data.bonding_curve.clone(),
            volume_change: parsed_data.volume_change,
            bonding_curve_info: parsed_data.bonding_curve_info.clone(),
            pool_info: parsed_data.pool_info.clone(),
            amount: parsed_data.amount,
            max_sol_cost: parsed_data.max_sol_cost,
            min_sol_output: parsed_data.min_sol_output,
            pool: parsed_data.pool.clone(),
            base_amount_in: parsed_data.base_amount_in,
            min_quote_amount_out: parsed_data.min_quote_amount_out,
            user_base_token_reserves: parsed_data.user_base_token_reserves,
            user_quote_token_reserves: parsed_data.user_quote_token_reserves,
            pool_base_token_reserves: parsed_data.pool_base_token_reserves,
            pool_quote_token_reserves: parsed_data.pool_quote_token_reserves,
            quote_amount_out: parsed_data.quote_amount_out,
            lp_fee_basis_points: parsed_data.lp_fee_basis_points,
            lp_fee: parsed_data.lp_fee,
            protocol_fee_basis_points: parsed_data.protocol_fee_basis_points,
            protocol_fee: parsed_data.protocol_fee,
            quote_amount_out_without_lp_fee: parsed_data.quote_amount_out_without_lp_fee,
            user_quote_amount_out: parsed_data.user_quote_amount_out,
            user_base_token_account: parsed_data.user_base_token_account.clone(),
            user_quote_token_account: parsed_data.user_quote_token_account.clone(),
            protocol_fee_recipient: parsed_data.protocol_fee_recipient.clone(),
            protocol_fee_recipient_token_account: parsed_data.protocol_fee_recipient_token_account.clone(),
            coin_creator: parsed_data.coin_creator.clone(),
            coin_creator_fee_basis_points: parsed_data.coin_creator_fee_basis_points,
            coin_creator_fee: parsed_data.coin_creator_fee,
            base_amount_out: parsed_data.base_amount_out,
            max_quote_amount_in: parsed_data.max_quote_amount_in,
        };
        let is_progressive_sell = self.is_progressive_sell;
        // Call execute_sell from copy_trading with progressive sell parameters
        match crate::engine::copy_trading::execute_sell(
            token_mint.to_string(),
            base_trade_info,
            self.app_state.clone(),
            self.swap_config.clone(),
            protocol.clone(),
            is_progressive_sell, // Use progressive sell
            Some(self.config.progressive_sell_chunks),
            Some(self.config.progressive_sell_interval * 1000), // Convert seconds to milliseconds
        ).await {
            Ok(_) => {
                self.logger.log(format!("Progressive sell completed for {}", token_mint).green().to_string());
                Ok(())
            },
            Err(e) => {
                self.logger.log(format!("Progressive sell failed: {}", e).red().to_string());
                Err(anyhow!("Progressive sell failed: {}", e))
            }
        }
    }

    /// Update metrics after a sell operation
    fn update_metrics_after_sell(&self, token_mint: &str, amount_sold: f64) -> Result<()> {
        // Get the necessary data and perform operations in a scoped block
        let should_remove = {
            let mut metrics_map = TOKEN_METRICS.lock().unwrap();
            
            // Check if we have metrics for this token
            let metrics = match metrics_map.get_mut(token_mint) {
                Some(m) => m,
                None => return Err(anyhow!("No metrics found for token {}", token_mint)),
            };
            
            // Update amount held
            metrics.amount_held -= amount_sold;
            
            // If we sold everything or almost everything (small dust amount), remove the metrics
            if metrics.amount_held <= 0.001 {
                // Almost empty, just remove it
                metrics_map.remove(token_mint);
                self.logger.log(format!("Removed token {} from portfolio after selling all", token_mint));
                
                // Also remove from tracking
                let mut tracking = TOKEN_TRACKING.lock().expect("Failed to lock token tracking");
                tracking.remove(token_mint);
                
                true
            } else {
                // Update cost basis proportionally
                if metrics.cost_basis > 0.0 {
                    let sold_percentage = amount_sold / (metrics.amount_held + amount_sold);
                    metrics.cost_basis *= 1.0 - sold_percentage;
                }
                
                self.logger.log(format!(
                    "Updated metrics for token {} after partial sell: amount_held={:.4}, cost_basis={:.4}", 
                    token_mint, metrics.amount_held, metrics.cost_basis
                ));
                
                false
            }
        }; // Lock is released here
        
        // Log the current token portfolio after changes
        self.log_token_portfolio();
        
        Ok(())
    }
    
    /// Send a transaction with priority fees
    async fn send_priority_transaction(
        &self,
        blockhash: Hash,
        keypair: &Keypair,
        instructions: Vec<Instruction>,
    ) -> Result<Signature> {
        self.logger.log("Sending transaction with priority fees".yellow().to_string());
        
        // Execute the transaction with priority
        let signatures = crate::core::tx::new_signed_and_send_nozomi(
            blockhash,
            keypair,
            instructions,
            &self.logger,
        ).await.map_err(|e| anyhow!("Transaction error: {}", e))?;
        
        if signatures.is_empty() {
            return Err(anyhow!("No transaction signature returned"));
        }
        
        let signature = Signature::from_str(&signatures[0])?;
        
        // Verify the transaction was successful
        self.logger.log("Verifying transaction...".yellow().to_string());
        
        // Wait for confirmation
        let mut retries = 5;
        while retries > 0 {
            match self.app_state.rpc_nonblocking_client.get_signature_status(&signature).await {
                Ok(Some(result)) => {
                    if result.is_ok() {
                        self.logger.log("Transaction confirmed".green().to_string());
                        return Ok(signature);
                    } else {
                        return Err(anyhow!("Transaction failed: {:?}", result));
                    }
                },
                Ok(None) => {
                    self.logger.log("Transaction not yet confirmed, waiting...".yellow().to_string());
                    sleep(Duration::from_millis(500)).await;
                    retries -= 1;
                },
                Err(e) => {
                    self.logger.log(format!("Error checking transaction status: {}", e).red().to_string());
                    sleep(Duration::from_millis(500)).await;
                    retries -= 1;
                }
            }
        }
        
        Err(anyhow!("Transaction verification timed out"))
    }
    
    /// Calculate the market cap of a token
    async fn calculate_market_cap(&self, token_mint: &str) -> Result<f64> {
        // Get token supply
        let token_supply = match self.app_state.rpc_nonblocking_client
            .get_token_supply(&Pubkey::from_str(token_mint)?)
            .await
        {
            Ok(supply) => supply.ui_amount.unwrap_or(0.0),
            Err(e) => return Err(anyhow!("Failed to get token supply: {}", e)),
        };
        
        // Get current price
        let price = self.get_current_price(token_mint).await?;
        
        // Calculate market cap
        let market_cap = token_supply * price;
        
        Ok(market_cap)
    }

    /// Calculate current price from trade info
    pub fn calculate_current_price(&self, trade_info: &TradeInfoFromToken) -> Option<f64> {
        match trade_info.dex_type {
            DexType::PumpFun => {
                // For Pump.fun: price = SOL amount / token amount
                trade_info.sol_amount.and_then(|sol| {
                    trade_info.token_amount.map(|tok| {
                        sol as f64 / tok as f64
                    })
                })
            },
            DexType::PumpSwap => {
                // For PumpSwap: price = quote reserves / base reserves
                trade_info.pool_quote_token_reserves.and_then(|quote| {
                    trade_info.pool_base_token_reserves.map(|base| {
                        quote as f64 / base as f64
                    })
                })
            },
            _ => None
        }
    }

    /// Calculate price impact for a trade
    pub fn calculate_price_impact(&self, trade_info: &TradeInfoFromToken) -> Option<f64> {
        match trade_info.dex_type {
            DexType::PumpFun => {
                // Price impact = (new virtual price - old virtual price) / old virtual price
                let old_price = trade_info.virtual_sol_reserves? as f64 / 
                               trade_info.virtual_token_reserves? as f64;
                let new_sol = (trade_info.virtual_sol_reserves? as i64 + 
                              trade_info.sol_amount? as i64) as u64;
                let new_tok = (trade_info.virtual_token_reserves? as i64 - 
                              trade_info.token_amount? as i64) as u64;
                
                if new_tok == 0 { return None; } // Avoid division by zero
                
                let new_price = new_sol as f64 / new_tok as f64;
                Some((new_price - old_price) / old_price)
            },
            DexType::PumpSwap => {
                // Similar calculation using pool reserves
                trade_info.pool_quote_token_reserves.and_then(|quote| {
                    trade_info.pool_base_token_reserves.and_then(|base| {
                        if base == 0 { return None; } // Avoid division by zero
                        
                        let old_price = quote as f64 / base as f64;
                        
                        // Calculate new reserves based on transaction type
                        let (new_quote, new_base) = if trade_info.is_buy {
                            // Buy: quote decreases, base increases
                            let quote_out = trade_info.quote_amount_out.unwrap_or(0);
                            let base_in = trade_info.base_amount_in.unwrap_or(0);
                            
                            ((quote as i64 - quote_out as i64) as u64, 
                             (base as i64 + base_in as i64) as u64)
                        } else {
                            // Sell: quote increases, base decreases
                            let quote_in = trade_info.quote_amount_out.unwrap_or(0); // Misleading name
                            let base_out = trade_info.base_amount_in.unwrap_or(0);   // Misleading name
                            
                            ((quote as i64 + quote_in as i64) as u64, 
                             (base as i64 - base_out as i64) as u64)
                        };
                        
                        if new_base == 0 { return None; } // Avoid division by zero
                        
                        let new_price = new_quote as f64 / new_base as f64;
                        Some((new_price - old_price) / old_price)
                    })
                })
            },
            _ => None
        }
    }
    
    /// Check liquidity conditions for selling
    pub fn check_liquidity_conditions(&self, trade_info: &TradeInfoFromToken) -> Option<String> {
        let current_liquidity = self.calculate_liquidity(trade_info)?;
        
        // Get metrics in a scoped block to ensure lock is released
        let entry_liquidity = {
            let metrics_lock = TOKEN_METRICS.lock().expect("Failed to lock token metrics for liquidity check");
            match metrics_lock.get(&trade_info.mint) {
                Some(metrics) => metrics.liquidity_at_entry,
                None => return None,
            }
        }; // Lock is released here
        
        // Only check if we have valid entry liquidity
        if entry_liquidity <= 0.0 {
            return None;
        }
        
        // Absolute liquidity threshold
        if current_liquidity < self.config.liquidity_monitor.min_absolute_liquidity {
            return Some(format!("Liquidity below absolute minimum: {} SOL", current_liquidity));
        }
        
        // Relative liquidity drop
        let liquidity_drop = (entry_liquidity - current_liquidity) / entry_liquidity;
        if liquidity_drop > self.config.liquidity_monitor.max_acceptable_drop {
            return Some(format!("Liquidity dropped {:.2}%", liquidity_drop * 100.0));
        }
        
        None
    }
    
    /// Calculate liquidity from trade info
    fn calculate_liquidity(&self, trade_info: &TradeInfoFromToken) -> Option<f64> {
        match trade_info.dex_type {
            DexType::PumpFun => {
                // For Pump.fun: liquidity = virtual SOL reserves in SOL units
                trade_info.virtual_sol_reserves.map(|v| v as f64 / 1_000_000_000.0)
            },
            DexType::PumpSwap => {
                // For PumpSwap: liquidity = min(base_reserves * price, quote_reserves) in SOL units
                let price = self.calculate_current_price(trade_info)?;
                let base_liquidity = trade_info.pool_base_token_reserves? as f64 * price;
                let quote_liquidity = trade_info.pool_quote_token_reserves? as f64 / 1_000_000_000.0;
                Some(base_liquidity.min(quote_liquidity))
            },
            _ => None
        }
    }
    
    /// Check volume conditions for selling
    pub fn check_volume_conditions(&self, trade_info: &TradeInfoFromToken) -> Option<String> {
        let current_volume = self.calculate_trade_volume(trade_info)?;
        
        // Update volume history and get stats, ensuring lock is released
        let volume_stats = {
            let mut metrics_lock = TOKEN_METRICS.lock().expect("Failed to lock token metrics for volume check");
            let metrics = match metrics_lock.get_mut(&trade_info.mint) {
                Some(m) => m,
                None => return None,
            };
            
            // Update volume history
            metrics.volume_history.push_back(current_volume);
            if metrics.volume_history.len() > self.config.volume_analysis.lookback_period {
                metrics.volume_history.pop_front();
            }
            
            // If we don't have enough data yet, return None
            if metrics.volume_history.len() < 5 {
                return None;
            }
            
            // Calculate volume moving average
            let volume_ma: f64 = metrics.volume_history.iter().sum::<f64>() / 
                            metrics.volume_history.len() as f64;
            
            // Return stats we need after releasing the lock
            (volume_ma, metrics.volume_history.len())
        }; // Lock is released here
        
        let (volume_ma, _history_len) = volume_stats;
        
        // Check volume spike
        if current_volume > volume_ma * self.config.volume_analysis.spike_threshold {
            return Some(format!(
                "Volume spike detected: current {:.2} SOL > {:.2}x average {:.2} SOL", 
                current_volume, 
                self.config.volume_analysis.spike_threshold,
                volume_ma
            ));
        }
        
        // Check volume drop
        if current_volume < volume_ma * self.config.volume_analysis.drop_threshold {
            return Some(format!(
                "Volume drop detected: current {:.2} SOL < {:.2}x average {:.2} SOL",
                current_volume,
                self.config.volume_analysis.drop_threshold,
                volume_ma
            ));
        }
        
        None
    }
    
    /// Calculate trade volume from trade info
    fn calculate_trade_volume(&self, trade_info: &TradeInfoFromToken) -> Option<f64> {
        match trade_info.dex_type {
            DexType::PumpFun => {
                // Volume in SOL terms
                trade_info.sol_amount.map(|v| v as f64 / 1_000_000_000.0)
            },
            DexType::PumpSwap => {
                // Volume in quote token (SOL) terms
                trade_info.quote_amount_out.map(|v| v as f64 / 1_000_000_000.0)
            },
            _ => None
        }
    }
    
    /// Check price conditions for selling (take profit, trailing stop, EMA crossover)
    pub fn check_price_conditions(&self, trade_info: &TradeInfoFromToken) -> Option<String> {
        let current_price = self.calculate_current_price(trade_info)?;
        
        // Update metrics and get necessary data in a scoped block
        let price_data = {
            let mut metrics_lock = TOKEN_METRICS.lock().expect("Failed to lock token metrics for price check");
            let metrics = match metrics_lock.get_mut(&trade_info.mint) {
                Some(m) => m,
                None => return None,
            };
            
            // Update price history
            metrics.price_history.push_back(current_price);
            if metrics.price_history.len() > 20 { // 20-period EMA
                metrics.price_history.pop_front();
            }
            
            // Update highest price
            if current_price > metrics.highest_price {
                metrics.highest_price = current_price;
            }
            
            // Update lowest price
            if metrics.lowest_price == 0.0 || current_price < metrics.lowest_price {
                metrics.lowest_price = current_price;
            }
            
            // Create a new VecDeque with the price history for EMA calculation
            let price_history_copy = metrics.price_history.clone();
            
            // Return the data we need after releasing the lock
            (
                metrics.entry_price,
                metrics.highest_price,
                price_history_copy.clone(),
                price_history_copy.len() >= 20
            )
        }; // Lock is released here
        
        let (entry_price, highest_price, price_history, has_enough_data) = price_data;
        
        // 1. Profit taking
        let pnl = (current_price - entry_price) / entry_price;
        if pnl >= self.config.profit_taking.target_percentage {
            return Some(format!("Take profit at {:.2}%", pnl * 100.0));
        }
        
        // 2. Trailing stop
        let drawdown_from_high = (highest_price - current_price) / highest_price;
        if pnl > 0.0 && drawdown_from_high >= self.config.trailing_stop.activation_percentage {
            return Some(format!(
                "Trailing stop triggered ({:.2}% from high)", 
                drawdown_from_high * 100.0
            ));
        }
        
        // 3. EMA crossover (only if we have enough data)
        if has_enough_data && self.check_ema_crossover(&price_history) {
            return Some("EMA crossover detected (9 EMA crossed below 20 EMA)".to_string());
        }
        
        None
    }
    
    /// Check time-based conditions for selling
    pub fn check_time_conditions(&self, trade_info: &TradeInfoFromToken) -> Option<String> {
        // Get current timestamp
        let current_timestamp = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(n) => n.as_secs(),
            Err(_) => return None,
        };
        
        // Get metrics in a scoped block
        let time_data = {
            let metrics_lock = TOKEN_METRICS.lock().expect("Failed to lock token metrics for time check");
            let metrics = match metrics_lock.get(&trade_info.mint) {
                Some(m) => m,
                None => return None,
            };
            
            // Return the data we need after releasing the lock
            (
                metrics.buy_timestamp,
                metrics.time_held,
                metrics.entry_price,
                metrics.current_price
            )
        }; // Lock is released here
        
        let (buy_timestamp, time_held, entry_price, current_price) = time_data;
        
        // Calculate time held
        let time_held_seconds = if buy_timestamp > 0 {
            current_timestamp.saturating_sub(buy_timestamp)
        } else {
            time_held
        };
        
        // Check max hold time
        if time_held_seconds >= self.config.time_based.max_hold_time_secs {
            return Some(format!("Max hold time exceeded: {}s >= {}s", 
                time_held_seconds, self.config.time_based.max_hold_time_secs));
        }
        
        // Calculate PNL
        let pnl = (current_price - entry_price) / entry_price;
        
        // For profitable trades, check minimum hold time
        if pnl > 0.0 && time_held_seconds < self.config.time_based.min_profit_time_secs {
            return None; // Don't sell profitable trades too early
        }
        
        None
    }
    
    /// Check if this might be wash trading (self-trading, circular trades)
    pub fn check_wash_trading(&self, trade_info: &TradeInfoFromToken) -> Option<String> {
        // Check if same address is both pool creator and trader
        if let Some(pool) = &trade_info.pool {
            if trade_info.user == *pool {
                return Some("Possible wash trading (user == pool)".to_string());
            }
        }
        
        // Check if price action looks manipulated
        // This is a simplified approach - in reality you'd need more sophisticated analysis
        if let (Some(current_price), Some(virtual_sol), Some(virtual_token)) = (
            self.calculate_current_price(trade_info),
            trade_info.virtual_sol_reserves,
            trade_info.virtual_token_reserves
        ) {
            let expected_price = virtual_sol as f64 / virtual_token as f64;
            let price_diff = (current_price - expected_price).abs() / expected_price;
            
            if price_diff > 0.1 { // 10% difference
                return Some(format!("Possible price manipulation: {:.2}% difference", price_diff * 100.0));
            }
        }
        
        None
    }
    
    /// Check large holder actions
    pub fn check_large_holder_actions(&self, trade_info: &TradeInfoFromToken) -> Option<String> {
        // Check if creator is selling
        if let Some(creator) = &trade_info.coin_creator {
            if trade_info.user == *creator && !trade_info.is_buy {
                return Some("Token creator selling".to_string());
            }
        }
        
        // Check for large wallet movements
        let trade_size = self.calculate_trade_volume(trade_info)?;
        let liquidity = self.calculate_liquidity(trade_info)?;
        
        if trade_size > liquidity * 0.1 { // 10% of liquidity
            return Some(format!("Large trade size detected: {:.2} SOL ({:.2}% of liquidity)",
                trade_size, (trade_size / liquidity) * 100.0));
        }
        
        None
    }
    
    /// Calculate EMA for price history
    fn calculate_ema(&self, prices: &VecDeque<f64>, period: usize) -> Option<f64> {
        if prices.len() < period {
            return None;
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];
        
        for price in prices.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }
        
        Some(ema)
    }
    
    /// Check if EMA crossover happened (9-period crosses below 20-period)
    fn check_ema_crossover(&self, prices: &VecDeque<f64>) -> bool {
        if prices.len() < 20 {
            return false;
        }
        
        // We need at least 2 data points to check for a crossover
        if prices.len() < 2 {
            return false;
        }
        
        // Calculate EMAs
        let ema9 = match self.calculate_ema(prices, 9) {
            Some(ema) => ema,
            None => return false,
        };
        
        let ema20 = match self.calculate_ema(prices, 20) {
            Some(ema) => ema,
            None => return false,
        };
        
        // Check for bearish crossover (9 EMA below 20 EMA)
        ema9 < ema20
    }
    
    /// Adjust strategy based on market conditions
    pub fn adjust_strategy_based_on_market(&mut self, market_condition: MarketCondition) {
        self.logger.log(format!("Adjusting strategy for market condition: {:?}", market_condition));
        
        match market_condition {
            MarketCondition::Bullish => {
                // Be more aggressive in taking profits
                self.config.profit_taking.target_percentage *= 1.2;
                self.config.trailing_stop.activation_percentage *= 1.2;
                self.logger.log("Adjusted for bullish market: increased profit targets".green().to_string());
            },
            MarketCondition::Bearish => {
                // Take profits earlier
                self.config.profit_taking.target_percentage *= 0.8;
                self.config.trailing_stop.activation_percentage *= 0.8;
                self.logger.log("Adjusted for bearish market: reduced profit targets".yellow().to_string());
            },
            MarketCondition::Volatile => {
                // Use tighter stops
                self.config.trailing_stop.trail_percentage *= 0.5;
                self.logger.log("Adjusted for volatile market: tightened stops".yellow().to_string());
            },
            MarketCondition::Stable => {
                // Let winners run longer
                self.config.profit_taking.target_percentage *= 1.5;
                self.logger.log("Adjusted for stable market: letting winners run longer".green().to_string());
            }
        }
    }
    
    /// Record trade execution for analytics
    pub async fn record_trade_execution(
        &self, 
        mint: &str, 
        reason: &str, 
        amount_sold: f64, 
        protocol: &str
    ) -> Result<()> {
        // Get current timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| anyhow!("Failed to get timestamp: {}", e))?
            .as_secs();
        
        // Get metrics for entry price
        let entry_price = {
            let metrics_map = TOKEN_METRICS.lock()
                .map_err(|e| anyhow!("Failed to lock token metrics: {}", e))?;
            
            metrics_map.get(mint)
                .map(|m| m.entry_price)
                .unwrap_or(0.0)
        };
        
        // Get current price
        let exit_price = match self.get_current_price(mint).await {
            Ok(price) => price,
            Err(_) => 0.0,
        };
        
        // Calculate PNL
        let pnl = if entry_price > 0.0 {
            ((exit_price - entry_price) / entry_price) * 100.0
        } else {
            0.0
        };
        
        // Create record
        let record = TradeExecutionRecord {
            mint: mint.to_string(),
            entry_price,
            exit_price,
            pnl,
            reason: reason.to_string(),
            timestamp,
            amount_sold,
            protocol: protocol.to_string(),
        };
        
        // Log record
        self.logger.log(format!(
            "Trade execution recorded: {} sold at {:.8} SOL (PNL: {:.2}%)",
            mint, exit_price, pnl
        ).green().to_string());
        
        // Add to history
        {
            let mut history = HISTORICAL_TRADES.lock()
                .expect("Failed to lock trade history for recording");
            
            history.push_back(record);
            
            // Keep history to a reasonable size
            if history.len() > 100 {
                history.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Monitor and sell based on trade info
    pub async fn monitor_and_sell(&self, trade_info: &TradeInfoFromToken) -> Result<bool> {
        self.logger.log(format!("Monitoring token: {}", trade_info.mint));
        
        // Update metrics with current price
        if let Some(price) = self.calculate_current_price(trade_info) {
            // Scope the lock to ensure it's released before any await points
            {
                let mut metrics_map = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
                if let Some(metrics) = metrics_map.get_mut(&trade_info.mint) {
                    metrics.current_price = price;
                    
                    // Update highest price if current price is higher
                    if price > metrics.highest_price {
                        metrics.highest_price = price;
                    }
                    
                    // Update time held
                    if let Some(timestamp) = Some(trade_info.timestamp) {
                        if metrics.buy_timestamp > 0 {
                            metrics.time_held = timestamp.saturating_sub(metrics.buy_timestamp);
                        }
                    }
                }
            } // Lock is released here
        }
        
        // Check all selling conditions
        let sell_reasons: Vec<String> = vec![
            self.check_liquidity_conditions(trade_info),
            self.check_volume_conditions(trade_info),
            self.check_price_conditions(trade_info),
            self.check_time_conditions(trade_info),
            self.check_wash_trading(trade_info),
            self.check_large_holder_actions(trade_info),
        ]
        .into_iter()
        .flatten()
        .collect();
        
        // If any conditions are met, execute sell
        if !sell_reasons.is_empty() {
            let reason = sell_reasons.join(", ");
            self.logger.log(format!("Sell conditions met: {}", reason).green().to_string());
            
            // Determine protocol based on token
            let protocol = match self.determine_best_protocol_for_token(&trade_info.mint).await {
                Ok(p) => p,
                Err(e) => {
                    self.logger.log(format!("Failed to determine protocol: {}", e).red().to_string());
                    return Err(anyhow!("Failed to determine protocol: {}", e));
                }
            };
            
            // Get enhanced TradeInfoFromToken for selling
            // We'll merge data from the original trade_info with additional data from metrics
            let enhanced_trade_info = match self.metrics_to_trade_info(&trade_info.mint, protocol.clone()).await {
                Ok(mut enhanced_info) => {
                    // Copy any available data from the original trade_info that might be useful
                    if enhanced_info.pool.is_none() && trade_info.pool.is_some() {
                        enhanced_info.pool = trade_info.pool.clone();
                    }
                    
                    if enhanced_info.pool_info.is_none() && trade_info.pool_info.is_some() {
                        enhanced_info.pool_info = trade_info.pool_info.clone();
                    }
                    
                    if enhanced_info.pool_base_token_reserves.is_none() && trade_info.pool_base_token_reserves.is_some() {
                        enhanced_info.pool_base_token_reserves = trade_info.pool_base_token_reserves;
                    }
                    
                    if enhanced_info.pool_quote_token_reserves.is_none() && trade_info.pool_quote_token_reserves.is_some() {
                        enhanced_info.pool_quote_token_reserves = trade_info.pool_quote_token_reserves;
                    }
                    
                    if enhanced_info.coin_creator.is_none() && trade_info.coin_creator.is_some() {
                        enhanced_info.coin_creator = trade_info.coin_creator.clone();
                    }
                    
                    enhanced_info
                },
                Err(e) => {
                    self.logger.log(format!("Failed to create enhanced trade info: {}", e).red().to_string());
                    return Err(anyhow!("Failed to create enhanced trade info: {}", e));
                }
            };
            
            // Execute progressive sell with the enhanced trade_info
            match self.progressive_sell(&trade_info.mint, &enhanced_trade_info, protocol).await {
                Ok(_) => {
                    self.logger.log(format!("Successfully sold token: {}", trade_info.mint).green().to_string());
                    
                    // Record the trade
                    if let Err(e) = self.record_trade_execution(
                        &trade_info.mint, 
                        &reason, 
                        0.0, // Amount will be filled in by execute_sell
                        &format!("{:?}", trade_info.dex_type)
                    ).await {
                        self.logger.log(format!("Failed to record trade: {}", e).red().to_string());
                    }
                    
                    return Ok(true);
                },
                Err(e) => {
                    self.logger.log(format!("Failed to sell token: {} - {}", trade_info.mint, e).red().to_string());
                    return Err(anyhow!("Failed to sell token: {}", e));
                }
            }
        }
        
        Ok(false)
    }
 
    
    /// Determine best protocol for selling a token
    async fn determine_best_protocol_for_token(&self, token_mint: &str) -> Result<SwapProtocol> {
        // Try PumpSwap first
        let pump_swap = PumpSwap::new(
            self.app_state.wallet.clone(),
            Some(self.app_state.rpc_client.clone()),
            Some(self.app_state.rpc_nonblocking_client.clone()),
        );
        
        match pump_swap.get_token_price(token_mint).await {
            Ok(_) => {
                self.logger.log(format!("Found token on PumpSwap: {}", token_mint).green().to_string());
                return Ok(SwapProtocol::PumpSwap);
            },
            Err(_) => {
                // Try PumpFun next
                let pump_fun = Pump::new(
                    self.app_state.rpc_nonblocking_client.clone(),
                    self.app_state.rpc_client.clone(),
                    self.app_state.wallet.clone(),
                );
                
                match pump_fun.get_token_price(token_mint).await {
                    Ok(_) => {
                        self.logger.log(format!("Found token on PumpFun: {}", token_mint).green().to_string());
                        return Ok(SwapProtocol::PumpFun);
                    },
                    Err(e) => {
                        return Err(anyhow!("Token not found on any supported DEX: {}", e));
                    }
                }
            }
        }
    }

    /// Convert TokenMetrics to a TradeInfoFromToken for analysis
    pub async fn metrics_to_trade_info(&self, token_mint: &str, protocol: SwapProtocol) -> Result<TradeInfoFromToken> {
        // Get a clone of the metrics to avoid holding the lock across await points
        let metrics = {
            let metrics_lock = TOKEN_METRICS.lock().expect("Failed to lock token metrics for conversion");
            match metrics_lock.get(token_mint) {
                Some(m) => m.clone(),
                None => return Err(anyhow!("No metrics found for token {}", token_mint)),
            }
        }; // MutexGuard is dropped here
        
        // Create timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| anyhow!("Failed to get current timestamp"))?
            .as_secs();
        
        // The amount of tokens we own
        let token_amount = metrics.amount_held;
        
        // Get a fresh blockhash
        let recent_blockhash = match self.app_state.rpc_nonblocking_client.get_latest_blockhash().await {
            Ok(hash) => hash,
            Err(e) => return Err(anyhow!("Failed to get blockhash: {}", e)),
        };
        
        // Create a DexType based on protocol
        let dex_type = match protocol {
            SwapProtocol::PumpSwap => DexType::PumpSwap,
            SwapProtocol::PumpFun => DexType::PumpFun,
            _ => return Err(anyhow!("Unsupported swap protocol")),
        };

        // Get pool and necessary reserves information from the blockchain
        let (
            pool,
            pool_info,
            pool_base_token_reserves,
            pool_quote_token_reserves,
            sol_amount,
            coin_creator
        ) = match protocol {
            SwapProtocol::PumpSwap => {
                let pump_swap = PumpSwap::new(
                    self.app_state.wallet.clone(),
                    Some(self.app_state.rpc_client.clone()),
                    Some(self.app_state.rpc_nonblocking_client.clone()),
                );
                
                let mint_pubkey = Pubkey::from_str(token_mint)
                    .map_err(|e| anyhow!("Invalid mint address: {}", e))?;
                
                match pump_swap.get_pump_swap_pool(token_mint).await {
                    Ok(pool_info) => {
                        // Convert price to estimated SOL amount
                        let est_sol_amount = (metrics.current_price * token_amount * 1_000_000_000.0) as u64;
                        
                        (
                            Some(pool_info.pool_id.to_string()),
                            Some(PoolInfo {
                                pool_id: pool_info.pool_id,
                                base_mint: pool_info.base_mint,
                                quote_mint: pool_info.quote_mint,
                                base_reserve: pool_info.base_reserve,
                                quote_reserve: pool_info.quote_reserve,
                                coin_creator: pool_info.coin_creator,
                            }),
                            Some(pool_info.base_reserve),
                            Some(pool_info.quote_reserve),
                            Some(est_sol_amount),
                            Some(pool_info.coin_creator.to_string())
                        )
                    },
                    Err(e) => {
                        self.logger.log(format!("Failed to get pool info: {}", e).red().to_string());
                        (None, None, None, None, None, None)
                    }
                }
            },
            SwapProtocol::PumpFun => {
                // For PumpFun, use the get_token_price method instead
                let pump_fun = Pump::new(
                    self.app_state.rpc_nonblocking_client.clone(),
                    self.app_state.rpc_client.clone(),
                    self.app_state.wallet.clone(),
                );
                
                match pump_fun.get_token_price(token_mint).await {
                    Ok(_price) => {
                        // We can't get complete token info, so use placeholders
                        // for reserves and estimate SOL amount
                        let est_sol_amount = (metrics.current_price * token_amount * 1_000_000_000.0) as u64;
                        
                        // Since we don't have actual reserves data, use reasonable defaults
                        let virtual_token_reserves = 1_000_000_000_000; // 1 trillion token units
                        let virtual_sol_reserves = (virtual_token_reserves as f64 * metrics.current_price) as u64;
                        
                        (
                            None, // PumpFun doesn't use pool field
                            None, // No pool_info for PumpFun
                            Some(virtual_token_reserves),
                            Some(virtual_sol_reserves),
                            Some(est_sol_amount),
                            None  // We don't have creator info
                        )
                    },
                    Err(e) => {
                        self.logger.log(format!("Failed to get token price: {}", e).red().to_string());
                        (None, None, None, None, None, None)
                    }
                }
            },
            _ => (None, None, None, None, None, None),
        };
        
        // Get user wallet to set as target
        let wallet_pubkey = self.app_state.wallet.pubkey().to_string();
        
        // Create TradeInfoFromToken with as much real information as possible
        Ok(TradeInfoFromToken {
            dex_type,
            mint: token_mint.to_string(),
            token_amount_f64: token_amount,
            timestamp,
            is_buy: false, // We're analyzing for sell
            slot: 0, // Not critical for selling
            recent_blockhash,
            signature: "metrics_to_trade_info".to_string(),
            target: wallet_pubkey.clone(),
            user: wallet_pubkey,
            
            // Include pool information if available
            pool,
            pool_info,
            pool_base_token_reserves,
            pool_quote_token_reserves,
            
            // Include estimated SOL amount based on current price
            sol_amount,
            
            // Include token amount in raw format if available
            token_amount: Some((token_amount * 1_000_000_000.0) as u64), // Assuming 9 decimals
            
            // Include creator info if available
            coin_creator: coin_creator,
            
            // Other fields can be None if not available
            min_quote_amount_out: None,
            user_base_token_reserves: None,
            user_quote_token_reserves: None,
            quote_amount_out: None,
            lp_fee_basis_points: None,
            lp_fee: None,
            protocol_fee_basis_points: None,
            protocol_fee: None,
            quote_amount_out_without_lp_fee: None,
            user_quote_amount_out: None,
            user_base_token_account: None,
            user_quote_token_account: None,
            protocol_fee_recipient: None,
            protocol_fee_recipient_token_account: None,
            coin_creator_fee_basis_points: None,
            coin_creator_fee: None,
            virtual_sol_reserves: pool_quote_token_reserves, // Reuse pool reserves for PumpSwap
            virtual_token_reserves: pool_base_token_reserves, // Reuse pool reserves for PumpSwap
            real_sol_reserves: None,
            real_token_reserves: None,
            bonding_curve: "".to_string(),
            volume_change: 0,
            bonding_curve_info: None,
            amount: None,
            max_sol_cost: None,
            min_sol_output: None,
            base_amount_in: None,
            base_amount_out: None,
            max_quote_amount_in: None,
        })
    }

    /// Run a backtest of the selling strategy on historical data
    pub async fn backtest_strategy(
        &self,
        historical_data: &[TradeInfoFromToken],
        selling_config: Option<SellingConfig>,
    ) -> Vec<TradeExecutionRecord> {
        let logger = Logger::new("[SELLING-STRATEGY-BACKTEST] => ".cyan().to_string());
        logger.log("Starting backtest of selling strategy...".to_string());
        
        // Create a new engine with the provided config or default
        let config = selling_config.unwrap_or_else(|| self.config.clone());
        let backtest_engine = SellingEngine {
            app_state: self.app_state.clone(),
            swap_config: self.swap_config.clone(),
            config,
            logger: Logger::new("[BACKTEST] => ".cyan().to_string()),
            token_manager: TokenManager::new(),
            is_progressive_sell: self.is_progressive_sell
        };
        
        // Clear global state for testing
        {
            let mut metrics = TOKEN_METRICS.lock().expect("Failed to lock token metrics");
            metrics.clear();
            
            let mut tracking = TOKEN_TRACKING.lock().expect("Failed to lock token tracking");
            tracking.clear();
            
            let mut history = HISTORICAL_TRADES.lock().expect("Failed to lock trade history");
            history.clear();
        }
        
        // Process all historical trades
        let mut tokens_bought: HashSet<String> = HashSet::new();
        
        for trade in historical_data {
            logger.log(format!("Processing trade: {} at timestamp {}", trade.mint, trade.timestamp));
            
            if trade.is_buy {
                // This is a buy - record it
                if !tokens_bought.contains(&trade.mint) {
                    // Calculate cost and amount
                    let token_amount = match trade.token_amount {
                        Some(amount) => amount as f64,
                        None => trade.token_amount_f64,
                    };
                    
                    let sol_amount = match trade.sol_amount {
                        Some(amount) => (amount as f64) / 1_000_000_000.0, // Convert from lamports
                        None => 0.0,
                    };
                    
                    if token_amount > 0.0 && sol_amount > 0.0 {
                        logger.log(format!("Recording buy: {} tokens for {} SOL", token_amount, sol_amount));
                        if let Err(e) = backtest_engine.record_buy(&trade.mint, token_amount, sol_amount) {
                            logger.log(format!("Error recording buy: {}", e).red().to_string());
                            continue; // Skip this trade if recording fails
                        }
                        tokens_bought.insert(trade.mint.clone());
                    }
                }
            } else {
                // This is a sell - evaluate selling condition
                if tokens_bought.contains(&trade.mint) {
                    match backtest_engine.monitor_and_sell(trade).await {
                        Ok(sold) => {
                            if sold {
                                logger.log(format!("Sold token: {}", trade.mint));
                                tokens_bought.remove(&trade.mint);
                            }
                        },
                        Err(e) => {
                            logger.log(format!("Error in backtest: {}", e).red().to_string());
                        }
                    }
                }
            }
        }
        
        // Get results
        let results: Vec<TradeExecutionRecord> = {
            let history: std::sync::MutexGuard<'_, VecDeque<TradeExecutionRecord>> = HISTORICAL_TRADES.lock().expect("Failed to lock trade history");
            history.clone().into_iter().collect()
        };
        
        // Log summary
        let mut total_profit = 0.0;
        let mut win_count = 0;
        let mut loss_count = 0;
        
        for record in &results {
            total_profit += record.pnl;
        
            // Count winning vs losing trades
            if record.pnl > 0.0 {
                win_count += 1;
            } else {
                loss_count += 1;
            }
        }
        
        logger.log(format!(
            "Backtest complete: {} trades, {} wins, {} losses, {:.2}% total return",
            results.len(),
            win_count,
            loss_count,
            total_profit * 100.0
        ));
        
        results
    }

    /// Generate backtest report from trade records
    pub fn generate_backtest_report(&self, records: &[TradeExecutionRecord]) -> String {
        let logger = Logger::new("[SELLING-STRATEGY-REPORT] => ".cyan().to_string());
        
        if records.is_empty() {
            return "No trades recorded in backtest.".to_string();
        }
        
        // Calculate statistics
        let mut total_pnl = 0.0;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut avg_win = 0.0;
        let mut avg_loss = 0.0;
        let mut max_win = 0.0;
        let mut max_loss = 0.0;
        let mut hold_times = Vec::new();
        let mut reason_counts = HashMap::new();
        
        let mut prev_timestamp = records[0].timestamp;
        
        for record in records {
            total_pnl += record.pnl;
            
            // Count winning vs losing trades
            if record.pnl > 0.0 {
                winning_trades += 1;
                avg_win += record.pnl;
                if record.pnl > max_win {
                    max_win = record.pnl;
                }
            } else {
                losing_trades += 1;
                avg_loss += record.pnl;
                if record.pnl < max_loss {
                    max_loss = record.pnl;
                }
            }
            
            // Calculate hold time
            if record.timestamp > prev_timestamp {
                hold_times.push(record.timestamp - prev_timestamp);
            }
            prev_timestamp = record.timestamp;
            
            // Count reasons
            let reason = record.reason.clone();
            *reason_counts.entry(reason).or_insert(0) += 1;
        }
        
        // Calculate averages
        avg_win = if winning_trades > 0 { avg_win / winning_trades as f64 } else { 0.0 };
        avg_loss = if losing_trades > 0 { avg_loss / losing_trades as f64 } else { 0.0 };
        
        // Calculate average hold time
        let avg_hold_time = if !hold_times.is_empty() {
            hold_times.iter().sum::<u64>() as f64 / hold_times.len() as f64
        } else {
            0.0
        };
        
        // Build report
        let mut report = String::new();
        report.push_str(&format!("Backtest Report\n"));
        report.push_str(&format!("==============\n\n"));
        report.push_str(&format!("Total Trades: {}\n", records.len()));
        report.push_str(&format!("Winning Trades: {} ({:.1}%)\n", 
            winning_trades, (winning_trades as f64 / records.len() as f64) * 100.0));
        report.push_str(&format!("Losing Trades: {} ({:.1}%)\n", 
            losing_trades, (losing_trades as f64 / records.len() as f64) * 100.0));
        report.push_str(&format!("Total PnL: {:.2}%\n", total_pnl * 100.0));
        report.push_str(&format!("Average Win: {:.2}%\n", avg_win * 100.0));
        report.push_str(&format!("Average Loss: {:.2}%\n", avg_loss * 100.0));
        report.push_str(&format!("Max Win: {:.2}%\n", max_win * 100.0));
        report.push_str(&format!("Max Loss: {:.2}%\n", max_loss * 100.0));
        report.push_str(&format!("Average Hold Time: {:.2} seconds\n", avg_hold_time));
        
        // Report on exit reasons
        report.push_str(&format!("\nExit Reasons:\n"));
        for (reason, count) in reason_counts.iter() {
            report.push_str(&format!("  {}: {} ({:.1}%)\n", 
                reason, 
                count, 
                (*count as f64 / records.len() as f64) * 100.0
            ));
        }
        
        // Log statistics
        logger.log(format!("Generated backtest report: {} trades, {:.2}% total return", 
            records.len(), total_pnl * 100.0));
        
        report
    }

    /// Analyze recent trades to determine market condition for dynamic strategy adjustment
    pub async fn analyze_market_condition(&self, recent_trades: &[TradeInfoFromToken]) -> MarketCondition {
        if recent_trades.is_empty() {
            return MarketCondition::Stable; // Default to stable if no data
        }
        
        // Extract prices from trades
        let mut prices: Vec<f64> = Vec::with_capacity(recent_trades.len());
        let mut volumes: Vec<f64> = Vec::with_capacity(recent_trades.len());
        let mut timestamps: Vec<u64> = Vec::with_capacity(recent_trades.len());
        
        for trade in recent_trades {
            // Calculate price from trade info
            if let Some(price) = self.calculate_current_price(trade) {
                prices.push(price);
            }
            
            // Extract volume
            if let Some(volume) = self.calculate_trade_volume(trade) {
                volumes.push(volume);
            }
            
            // Extract timestamp
            timestamps.push(trade.timestamp);
        }
        
        // Sort by timestamp to ensure chronological order
        let mut price_time_pairs: Vec<(u64, f64)> = timestamps.iter()
            .zip(prices.iter())
            .map(|(t, p)| (*t, *p))
            .collect();
        price_time_pairs.sort_by_key(|(t, _)| *t);
        
        // Re-extract sorted prices
        let sorted_prices: Vec<f64> = price_time_pairs.iter()
            .map(|(_, p)| *p)
            .collect();
        
        // Calculate time periods between price points
        // Convert to u64 during the mapping process
        let _time_periods: Vec<u64> = if price_time_pairs.len() >= 2 {
            price_time_pairs.windows(2)
                .map(|w| w[1].0.saturating_sub(w[0].0)) // Using timestamp differences
                .collect()
        } else {
            vec![0] // Default if not enough data
        };
        
        // Price volatility (std deviation / mean)
        let price_volatility = if !sorted_prices.is_empty() {
            let mean_price = sorted_prices.iter().sum::<f64>() / sorted_prices.len() as f64;
            let variance = sorted_prices.iter()
                .map(|p| (p - mean_price).powi(2))
                .sum::<f64>() / sorted_prices.len() as f64;
            (variance.sqrt() / mean_price).abs()
        } else {
            0.0
        };
        
        // Volume volatility
        let volume_volatility = if !volumes.is_empty() {
            let mean_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
            let variance = volumes.iter()
                .map(|v| (v - mean_volume).powi(2))
                .sum::<f64>() / volumes.len() as f64;
            (variance.sqrt() / mean_volume).abs()
        } else {
            0.0
        };
        
        // Price trend (positive = up, negative = down)
        let price_trend = if sorted_prices.len() >= 2 {
            (sorted_prices[sorted_prices.len() - 1] - sorted_prices[0]) / sorted_prices[0]
        } else {
            0.0
        };
        
        // Log analysis results
        self.logger.log(format!(
            "Market analysis: Volatility: {:.2}%, Trend: {:.2}%, Volume vol: {:.2}%",
            price_volatility * 100.0, price_trend * 100.0, volume_volatility * 100.0
        ).blue().to_string());
        
        // Determine market condition based on analysis
        if price_volatility > 0.15 {
            // High volatility market
            if price_trend > 0.05 {
                self.logger.log("Market condition: Bullish with high volatility".green().to_string());
                MarketCondition::Bullish
            } else if price_trend < -0.05 {
                self.logger.log("Market condition: Bearish with high volatility".red().to_string());
                MarketCondition::Bearish
            } else {
                self.logger.log("Market condition: Volatile with no clear trend".yellow().to_string());
                MarketCondition::Volatile
            }
        } else {
            // Low volatility market
            if price_trend > 0.05 {
                self.logger.log("Market condition: Stable uptrend".green().to_string());
                MarketCondition::Bullish
            } else if price_trend < -0.05 {
                self.logger.log("Market condition: Stable downtrend".red().to_string());
                MarketCondition::Bearish
            } else {
                self.logger.log("Market condition: Stable sideways".blue().to_string());
                MarketCondition::Stable
            }
        }
    }

    /// Execute progressive sell for token
    pub async fn execute_progressive_sell(&self, token_mint: &str) -> Result<()> {
        // Determine protocol based on token
        let protocol = self.determine_best_protocol_for_token(token_mint).await?;
        
        // Get TradeInfoFromToken with all the required parameters
        let trade_info = self.metrics_to_trade_info(token_mint, protocol.clone()).await?;
        
        // Execute progressive sell with properly populated trade_info
        self.progressive_sell(token_mint, &trade_info, protocol).await
    }
} 
