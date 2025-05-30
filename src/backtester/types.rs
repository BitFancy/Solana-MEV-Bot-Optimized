// src/backtester/types.rs

use serde::{Serialize, Deserialize};

/// Represents a single simulated trade executed during a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedTrade {
    pub entry_time_ms: u64,
    pub exit_time_ms: u64, // Can be same as entry if considered instantaneous
    pub path_id: String,     // Identifier for the strategy or path that triggered the trade
    pub entry_price: f64,  // Simulated entry price for the asset or basket
    pub exit_price: f64,   // Simulated exit price
    pub profit: f64,         // Profit or loss from this trade (absolute value)
    pub action_taken: String, // Describes why the trade was made (e.g., "AI_Execute", "RL_Signal")
    pub balance_after_trade: f64,
}

/// Summarizes the performance results of a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub pair_address: String,
    pub exchange: String,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
    pub initial_balance: f64,
    pub final_balance: f64,
    pub total_trades: usize,
    pub profitable_trades: usize,
    pub total_profit: f64, // Sum of all profits and losses
    pub profit_factor: Option<f64>, // Gross profit / Gross loss
    pub win_rate: f64, // Profitable trades / Total trades
    pub average_profit_per_trade: f64,
    pub max_drawdown: f64, // Placeholder - calculation can be complex
    pub sharpe_ratio: f64, // Placeholder - calculation requires risk-free rate and std dev of returns
    pub trade_history: Vec<SimulatedTrade>,
}

impl BacktestReport {
    pub fn new(
        pair_address: String,
        exchange: String,
        start_time_ms: u64,
        end_time_ms: u64,
        initial_balance: f64,
        final_balance: f64,
        simulated_trades: Vec<SimulatedTrade>,
    ) -> Self {
        let total_trades = simulated_trades.len();
        let profitable_trades = simulated_trades.iter().filter(|t| t.profit > 0.0).count();
        let total_profit: f64 = simulated_trades.iter().map(|t| t.profit).sum();

        let win_rate = if total_trades > 0 { profitable_trades as f64 / total_trades as f64 } else { 0.0 };
        let average_profit_per_trade = if total_trades > 0 { total_profit / total_trades as f64 } else { 0.0 };

        let gross_profit: f64 = simulated_trades.iter().filter(|t| t.profit > 0.0).map(|t| t.profit).sum();
        let gross_loss: f64 = simulated_trades.iter().filter(|t| t.profit < 0.0).map(|t| t.profit.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { Some(gross_profit / gross_loss) } else if gross_profit > 0.0 { Some(f64::INFINITY) } else { None };


        // Max drawdown and Sharpe ratio are complex and are placeholders for now.
        // Max drawdown requires iterating through balance history to find peak-to-trough declines.
        // Sharpe ratio requires periodic returns, their standard deviation, and a risk-free rate.
        let max_drawdown = 0.0; // Placeholder
        let sharpe_ratio = 0.0; // Placeholder

        Self {
            pair_address,
            exchange,
            start_time_ms,
            end_time_ms,
            initial_balance,
            final_balance,
            total_trades,
            profitable_trades,
            total_profit,
            profit_factor,
            win_rate,
            average_profit_per_trade,
            max_drawdown,
            sharpe_ratio,
            trade_history: simulated_trades,
        }
    }
}
