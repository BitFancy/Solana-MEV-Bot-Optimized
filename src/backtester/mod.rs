// src/backtester/mod.rs

pub mod config;
pub mod types;
pub mod engine;

pub use config::BacktestConfig;
pub use types::{BacktestReport, SimulatedTrade};
pub use engine::BacktestEngine;

/// Initializes the backtesting module (conceptual).
/// This could pre-load any shared resources or configurations for backtesting.
pub fn initialize_backtester() {
    log::info!("Backtester module initialized. (Conceptual)");
}
