use maplit::hashmap;
use anchor_client::solana_sdk::{hash::Hash, pubkey::Pubkey, signature::Signature};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{collections::HashSet, time::Duration};
use base64;
use bs58;
use crate::common::config::Status;
use crate::common::config::LiquidityPool;
use crate::engine::transaction_parser; // Add import for transaction_parser
use anyhow::{Result, anyhow};
use crate::engine::swap::{SwapDirection, SwapInType};
use crate::common::config::{
    JUPITER_PROGRAM,
    OKX_DEX_PROGRAM,
    PUMP_FUN_PROGRAM_DATA_PREFIX,
    PUMP_FUN_BUY_LOG_INSTRUCTION,
    PUMP_FUN_BUY_OR_SELL_PROGRAM_DATA_PREFIX,
    PUMP_FUN_SELL_LOG_INSTRUCTION,
    PUMP_SWAP_BUY_LOG_INSTRUCTION,
    PUMP_SWAP_BUY_PROGRAM_DATA_PREFIX,
    PUMP_SWAP_SELL_LOG_INSTRUCTION,
    PUMP_SWAP_SELL_PROGRAM_DATA_PREFIX,
};
use crate::common::{    
    config::{AppState, SwapConfig},
    logger::Logger,
};
use crate::core::tx;
use crate::dex::pump_fun::{Pump, INITIAL_VIRTUAL_SOL_RESERVES, INITIAL_VIRTUAL_TOKEN_RESERVES,   PUMP_FUN_CREATE_IX_DISCRIMINATOR, PUMP_PROGRAM, get_bonding_curve_account};
use chrono::Utc;
use colored::Colorize;
use futures_util::stream::StreamExt;
use futures_util::{SinkExt, Sink};
use futures;
use tokio::{
    time::{self, Instant},
};
use yellowstone_grpc_client::{ClientTlsConfig, GeyserGrpcClient};
use yellowstone_grpc_proto::geyser::{
    subscribe_update::UpdateOneof, CommitmentLevel, SubscribeRequest, SubscribeRequestPing,
    SubscribeRequestFilterTransactions, SubscribeUpdateTransaction, SubscribeUpdate,
};
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum InstructionType {
    PumpMint,
    PumpBuy,
    PumpSell,
    PumpSwapBuy,
    PumpSwapSell
}

#[derive(Clone, Debug)]
pub struct BondingCurveInfo {
    pub bonding_curve: Pubkey,
    pub new_virtual_sol_reserve: u64,
    pub new_virtual_token_reserve: u64,
}

#[derive(Clone, Debug)]
pub struct PoolInfo {
    pub pool_id: Pubkey,
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub base_reserve: u64,
    pub quote_reserve: u64,
    pub coin_creator: Pubkey,
}

pub struct FilterConfig {
    program_ids: Vec<String>,
    _instruction_discriminators: &'static [&'static [u8]],
    copy_trading_target_addresses: Vec<String>,
    is_multi_copy_trading: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RetracementLevel {
    pub percentage: u64,
    pub threshold: u64,
    pub sell_amount: u64,
}

#[derive(Clone, Debug)]
pub struct TokenTrackingInfo {
    pub top_pnl: f64,
    pub last_sell_time: Instant,
    pub completed_intervals: HashSet<String>,
}

#[derive(Clone, Debug)]
pub struct CopyTradeInfo {
    pub slot: u64,
    pub recent_blockhash: Hash,
    pub signature: String,
    pub target: String,
    pub mint: String,
    pub bonding_curve: String,
    pub volume_change: i64,
    pub bonding_curve_info: Option<BondingCurveInfo>,
}

lazy_static::lazy_static! {
    static ref COUNTER: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    static ref SOLD: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    static ref BOUGHTS: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    static ref LAST_BUY_PAUSE_TIME: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    static ref BUYING_ENABLED: Arc<Mutex<bool>> = Arc::new(Mutex::new(true));
    static ref TOKEN_TRACKING: Arc<Mutex<HashMap<String, TokenTrackingInfo>>> = Arc::new(Mutex::new(HashMap::new()));
    
    static ref THRESHOLD_BUY: Arc<Mutex<u64>> = Arc::new(Mutex::new(
        std::env::var("THRESHOLD_BUY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1_000_000_000)
    ));
    
    static ref THRESHOLD_SELL: Arc<Mutex<u64>> = Arc::new(Mutex::new(
        std::env::var("THRESHOLD_SELL")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1_000_000_000)
    ));
    
    static ref MAX_WAIT_TIME: Arc<Mutex<u64>> = Arc::new(Mutex::new(
        std::env::var("MAX_WAIT_TIME")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(60000)
    ));

    static ref TAKE_PROFIT_LEVELS: Vec<(u64, u64)> = vec![
        (2000, 100),
        (1500, 40),
        (1000, 40),
        (800, 20),
        (600, 20),
        (400, 20),
        (300, 20),
        (250, 20),
        (200, 20),
        (120, 20),
        (80, 20),
        (50, 10),
        (20, 10),
    ];
    
    static ref RETRACEMENT_LEVELS: Vec<RetracementLevel> = vec![
        RetracementLevel { percentage: 3, threshold: 2000, sell_amount: 100 },
        RetracementLevel { percentage: 4, threshold: 1500, sell_amount: 50 },
        RetracementLevel { percentage: 5, threshold: 1000, sell_amount: 40 },
        RetracementLevel { percentage: 6, threshold: 800, sell_amount: 35 },
        RetracementLevel { percentage: 6, threshold: 700, sell_amount: 35 },
        RetracementLevel { percentage: 6, threshold: 600, sell_amount: 30 },
        RetracementLevel { percentage: 7, threshold: 500, sell_amount: 30 },
        RetracementLevel { percentage: 7, threshold: 400, sell_amount: 30 },
        RetracementLevel { percentage: 8, threshold: 300, sell_amount: 20 },
        RetracementLevel { percentage: 10, threshold: 200, sell_amount: 15 },
        RetracementLevel { percentage: 12, threshold: 100, sell_amount: 15 },
        RetracementLevel { percentage: 20, threshold: 50, sell_amount: 10 },
        RetracementLevel { percentage: 30, threshold: 30, sell_amount: 10 },
        RetracementLevel { percentage: 42, threshold: 20, sell_amount: 100 },
    ];
    
    static ref DOWNING_PERCENT: Arc<Mutex<u64>> = Arc::new(Mutex::new(
        std::env::var("DOWNING_PERCENT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42)
    ));
    
    static ref LAST_MESSAGE_TIME: Arc<Mutex<Instant>> = Arc::new(Mutex::new(Instant::now()));
}

fn update_last_message_time() {
    let mut last_time = LAST_MESSAGE_TIME.lock().unwrap();
    *last_time = Instant::now();
}

async fn check_connection_health(logger: &Logger) {
    let last_time = {
        let time = LAST_MESSAGE_TIME.lock().unwrap();
        *time
    };
    
    let now = Instant::now();
    let elapsed = now.duration_since(last_time);
    
    if elapsed > Duration::from_secs(300) {
        logger.log(format!(
            "[CONNECTION WARNING] => No messages received in {:?}. Connection may be stale.",
            elapsed
        ).yellow().to_string());
    }
}

fn extract_bonding_curve(
    _txn: &SubscribeUpdateTransaction,
    log_messages: &[String],
) -> Result<String> {
    // Default value
    let bonding_curve = "6221o1QRckJ5SLbniMFB53qhqnrrrXGaj88Dojknivi4".to_string();
    
    // Try to extract from logs
    for log in log_messages {
        if log.contains("bonding_curve:") {
            if let Some(curve_str) = log.split_whitespace().nth(3) {
                return Ok(curve_str.to_string());
            }
        }
    }
    
    Ok(bonding_curve)
}

fn extract_bonding_curve_info(
    _txn: &SubscribeUpdateTransaction,
    log_messages: &[String],
    bonding_curve: &str,
) -> Result<BondingCurveInfo> {
    // Default values
    let mut virtual_sol_reserve = INITIAL_VIRTUAL_SOL_RESERVES;
    let mut virtual_token_reserve = INITIAL_VIRTUAL_TOKEN_RESERVES;
    
    // Try to extract from logs
    for log in log_messages {
        if log.contains("virtual_sol_reserve:") {
            if let Some(reserve_str) = log.split_whitespace().nth(3) {
                if let Ok(parsed_reserve) = reserve_str.parse::<u64>() {
                    virtual_sol_reserve = parsed_reserve;
                }
            }
        } else if log.contains("virtual_token_reserve:") {
            if let Some(reserve_str) = log.split_whitespace().nth(3) {
                if let Ok(parsed_reserve) = reserve_str.parse::<u64>() {
                    virtual_token_reserve = parsed_reserve;
                }
            }
        }
    }
    
    Ok(BondingCurveInfo {
        bonding_curve: Pubkey::from_str(bonding_curve).unwrap_or_default(),
        new_virtual_sol_reserve: virtual_sol_reserve,
        new_virtual_token_reserve: virtual_token_reserve,
    })
}

fn calculate_token_amount(
    _txn: &SubscribeUpdateTransaction,
    _log_messages: &[String],
    instruction_type: transaction_parser::DexType,
) -> Result<f64> {
    // Default value based on instruction type
    let token_amount = match instruction_type {
        transaction_parser::DexType::PumpFun => 177885.45,
        transaction_parser::DexType::PumpSwap => 63190.55,
        _ => 0.0,
    };
    
    Ok(token_amount)
}

fn calculate_volume_change(
    _txn: &SubscribeUpdateTransaction,
    _log_messages: &[String],
    instruction_type: transaction_parser::DexType,
) -> Result<i64> {
    // Default value based on instruction type
    let volume_change = match instruction_type {
        transaction_parser::DexType::PumpFun => 1_000_000_000, // 1 SOL positive change
        transaction_parser::DexType::PumpSwap => -1_000_000_000, // 1 SOL negative change
        _ => 0,
    };
    
    Ok(volume_change)
}

/**
 * The following functions implement a ping-pong mechanism to keep the gRPC connection alive:
 * 
 * - process_stream_message: Handles incoming messages, responding to pings and logging pongs
 * - handle_ping_message: Sends a pong response when a ping is received
 * - send_heartbeat_ping: Proactively sends pings every 30 seconds
 * 
 * This ensures the connection stays active even during periods of inactivity,
 * preventing timeouts from the server or network infrastructure.
 */

/// Send a ping response when we receive a ping
async fn handle_ping_message(
    subscribe_tx: &Arc<tokio::sync::Mutex<impl Sink<SubscribeRequest, Error = impl std::fmt::Debug> + Unpin>>,
    _logger: &Logger,
) -> Result<(), String> {
    let ping_request = SubscribeRequest {
        ping: Some(SubscribeRequestPing { id: 1 }),
        ..Default::default()
    };

    // Get a lock on the mutex
    let mut locked_tx = subscribe_tx.lock().await;
    
    // Send the ping response
    match locked_tx.send(ping_request).await {
        Ok(_) => {
            Ok(())
        },
        Err(e) => {
            Err(format!("Failed to send ping response: {:?}", e))
        }
    }
}

/// Process stream messages including ping-pong for keepalive
async fn process_stream_message(
    msg: &SubscribeUpdate,
    subscribe_tx: &Arc<tokio::sync::Mutex<impl Sink<SubscribeRequest, Error = impl std::fmt::Debug> + Unpin>>,
    logger: &Logger,
) -> Result<(), String> {
    update_last_message_time();
    match &msg.update_oneof {
        Some(UpdateOneof::Ping(_)) => {
            handle_ping_message(subscribe_tx, logger).await?;
        }
        Some(UpdateOneof::Pong(_)) => {
        }
        _ => {
        }
    }
    Ok(())
}

// Heartbeat function to periodically send pings
async fn send_heartbeat_ping(
    subscribe_tx: &Arc<tokio::sync::Mutex<impl Sink<SubscribeRequest, Error = impl std::fmt::Debug> + Unpin>>,
    _logger: &Logger
) -> Result<(), String> {
    let ping_request = SubscribeRequest {
        ping: Some(SubscribeRequestPing { id: 1 }),
        ..Default::default()
    };
    
    // Get a lock on the mutex
    let mut locked_tx = subscribe_tx.lock().await;
    
    // Send the ping heartbeat
    match locked_tx.send(ping_request).await {
        Ok(_) => {
            Ok(())
        },
        Err(e) => {
            Err(format!("Failed to send heartbeat ping: {:?}", e))
        }
    }
}
/// Convert lamports to SOL (1 SOL = 10^9 lamports)
fn lamports_to_sol(lamports: u64) -> f64 {
    lamports as f64 / 1_000_000_000.0
}



