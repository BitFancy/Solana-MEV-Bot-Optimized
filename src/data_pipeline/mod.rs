// src/data_pipeline/mod.rs

pub mod types;
pub mod fetchers;
pub mod storage;

// Re-export key structs and functions for easier access from outside the data_pipeline module.
pub use types::{
    RawMarketEvent,
    ProcessedMarketData,
    MarketEventType,
};

pub use fetchers::{
    connect_to_realtime_stream,
    fetch_historical_data,
    get_supported_pairs,
};

pub use storage::{
    save_raw_market_event,
    save_processed_market_data,
    retrieve_historical_data,
};

// Optional: Define any high-level functions for the data pipeline orchestration here.
// For example, a function that fetches, processes, and stores data.

use log::error;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;
use tokio::sync::mpsc;

/// Initializes the data pipeline (conceptual).
/// This could involve setting up database connections, and starting real-time data fetchers.
pub async fn initialize_data_pipeline() {
    log::info!("Data Pipeline initializing... (Conceptual)");

    // Example: Start a real-time stream for a specific account
    // In a real application, these would come from config
    let rpc_ws_url = "wss://api.mainnet-beta.solana.com".to_string(); // Example public RPC
    // Replace with an actual account address you want to monitor, e.g., a busy market or token account
    let account_to_watch_str = "So11111111111111111111111111111111111111112"; // Wrapped SOL mint (example)

    match Pubkey::from_str(account_to_watch_str) {
        Ok(account_pubkey) => {
            // Create an MPSC channel for this stream
            let (tx, mut rx) = mpsc::channel::<RawMarketEvent>(100); // Buffer size 100

            // Spawn the stream connector in a new Tokio task
            tokio::spawn(async move {
                if let Err(e) = fetchers::connect_to_realtime_stream(rpc_ws_url, account_pubkey, tx).await {
                    error!("Real-time stream for account {} failed: {}", account_pubkey, e);
                }
            });

            // Spawn a consumer task for the received data (example)
            tokio::spawn(async move {
                log::info!("Consumer task started for account {}.", account_pubkey);
                while let Some(event) = rx.recv().await {
                    log::info!("[Consumer] Received RawMarketEvent: Pair: {}, Type: {:?}, Timestamp: {}, Data: '{}'",
                        event.pair_address, event.event_type, event.timestamp_ms, event.raw_data);
                    // Here, the event could be passed to storage, processing, MarketPredictor, etc.
                    // For example, saving it:
                    // if let Err(e) = storage::save_raw_market_event(&event).await {
                    //     error!("Failed to save raw market event: {}", e);
                    // }
                }
                log::info!("Consumer task ended for account {}.", account_pubkey);
            });

            log::info!("Successfully launched real-time data stream and consumer for account {}.", account_pubkey);

        }
        Err(e) => {
            error!("Failed to parse Pubkey for account to watch '{}': {}", account_to_watch_str, e);
        }
    }

    // In a real application, this might also:
    // - Establish database connections
    // - Initialize schema if it doesn't exist
    log::info!("Data Pipeline initialization tasks (conceptual) completed.");
}
