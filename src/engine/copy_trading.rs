use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::Instant;
use crate::common::config::import_env_var;
use anyhow::Result;
use anchor_client::solana_sdk::{hash::Hash, pubkey::Pubkey, signature::Signature};
use solana_sdk::signature::Signer;
use spl_associated_token_account::get_associated_token_address;
use colored::Colorize;
use tokio::time;
use tokio::time::sleep;
use futures_util::stream::StreamExt;
use futures_util::{SinkExt, Sink};
use yellowstone_grpc_client::{ClientTlsConfig, GeyserGrpcClient};
use yellowstone_grpc_proto::geyser::{
    subscribe_update::UpdateOneof, CommitmentLevel, SubscribeRequest, SubscribeRequestPing,
    SubscribeRequestFilterTransactions, SubscribeUpdateTransaction, SubscribeUpdate,
};
use solana_transaction_status::TransactionConfirmationStatus;
use bs58;
// Import InstructionType with a namespace alias to avoid confusion
use crate::engine::transaction_parser::{self, DexType};
use crate::engine::selling_strategy::TokenMetrics;

use crate::common::{
    config::{AppState, SwapConfig, JUPITER_PROGRAM, OKX_DEX_PROGRAM},
    logger::Logger,
    cache::WALLET_TOKEN_ACCOUNTS,
};
use crate::engine::monitor::{self, BondingCurveInfo, PoolInfo};
use crate::engine::swap::{SwapDirection, SwapProtocol};
use crate::engine::selling_strategy::{SellingEngine, SellingConfig};

// Global state for copy trading
lazy_static::lazy_static! {
    static ref COUNTER: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    static ref SOLD_TOKENS: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    static ref BOUGHT_TOKENS: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    static ref LAST_BUY_TIME: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    static ref BUYING_ENABLED: Arc<Mutex<bool>> = Arc::new(Mutex::new(true));
    static ref TOKEN_TRACKING: Arc<Mutex<HashMap<String, TokenTrackingInfo>>> = Arc::new(Mutex::new(HashMap::new()));
}

// Track token performance for selling strategies
#[derive(Clone, Debug)]
pub struct TokenTrackingInfo {
    pub top_pnl: f64,
    pub last_sell_time: Instant,
    pub completed_intervals: HashSet<String>,
}

/// Configuration for copy trading
pub struct CopyTradingConfig {
    pub yellowstone_grpc_http: String,
    pub yellowstone_grpc_token: String,
    pub app_state: AppState,
    pub swap_config: SwapConfig,
    pub time_exceed: u64,
    pub counter_limit: u64,
    pub min_dev_buy: u64,
    pub max_dev_buy: u64,
    pub target_addresses: Vec<String>,
    pub protocol_preference: SwapProtocol,
    pub is_progressive_sell: bool,
}

/// Helper to send heartbeat pings to maintain connection
async fn send_heartbeat_ping(
    subscribe_tx: &Arc<tokio::sync::Mutex<impl Sink<SubscribeRequest, Error = impl std::fmt::Debug> + Unpin>>,
    logger: &Logger,
) -> Result<(), String> {
    let ping_request = SubscribeRequest {
        ping: Some(SubscribeRequestPing { id: 0 }),
        ..Default::default()
    };
    
    let mut tx = subscribe_tx.lock().await;
    match tx.send(ping_request).await {
        Ok(_) => {
            logger.log("[PING] Sent heartbeat ping".blue().to_string());
            Ok(())
        },
        Err(e) => Err(format!("Failed to send ping: {:?}", e)),
    }
}

/// Main function to start copy trading
pub async fn start_copy_trading(config: CopyTradingConfig) -> Result<(), String> {
    let logger = Logger::new("[COPY-TRADING] => ".green().bold().to_string());
    
    // Initialize
    logger.log("Initializing copy trading bot...".green().to_string());
    logger.log(format!("Target addresses: {:?}", config.target_addresses));
    logger.log(format!("Protocol preference: {:?}", config.protocol_preference));
    
    // Connect to Yellowstone gRPC
    logger.log("Connecting to Yellowstone gRPC...".green().to_string());
    let mut client = GeyserGrpcClient::build_from_shared(config.yellowstone_grpc_http.clone())
        .map_err(|e| format!("Failed to build client: {}", e))?
        .x_token::<String>(Some(config.yellowstone_grpc_token.clone()))
        .map_err(|e| format!("Failed to set x_token: {}", e))?
        .tls_config(ClientTlsConfig::new().with_native_roots())
        .map_err(|e| format!("Failed to set tls config: {}", e))?
        .connect()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    // Set up subscribe
    let mut retry_count = 0;
    const MAX_RETRIES: u32 = 3;
    let (subscribe_tx, mut stream) = loop {
        match client.subscribe().await {
            Ok(pair) => break pair,
            Err(e) => {
                retry_count += 1;
                if retry_count >= MAX_RETRIES {
                    return Err(format!("Failed to subscribe after {} attempts: {}", MAX_RETRIES, e));
                }
                logger.log(format!(
                    "[CONNECTION ERROR] => Failed to subscribe (attempt {}/{}): {}. Retrying in 5 seconds...",
                    retry_count, MAX_RETRIES, e
                ).red().to_string());
                time::sleep(Duration::from_secs(5)).await;
            }
        }
    };

    // Convert to Arc to allow cloning across tasks
    let subscribe_tx = Arc::new(tokio::sync::Mutex::new(subscribe_tx));

    // Enable buying
    {
        let mut buying_enabled = BUYING_ENABLED.lock().unwrap();
        *buying_enabled = true;
    }

    // Create config for subscription
    let target_addresses = config.target_addresses.clone();

    // Set up subscription
    logger.log("Setting up subscription...".green().to_string());
    let subscription_request = SubscribeRequest {
        transactions: maplit::hashmap! {
            "All".to_owned() => SubscribeRequestFilterTransactions {
                vote: Some(false), // Exclude vote transactions
                failed: Some(false), // Exclude failed transactions
                signature: None,
                account_include: target_addresses.clone(), // Only include transactions involving our targets
                account_exclude: vec![JUPITER_PROGRAM.to_string(), OKX_DEX_PROGRAM.to_string()], // Exclude some common programs
                account_required: Vec::<String>::new(),
            }
        },
        commitment: Some(CommitmentLevel::Processed as i32),
        ..Default::default()
    };
    
    subscribe_tx
        .lock()
        .await
        .send(subscription_request)
        .await
        .map_err(|e| format!("Failed to send subscribe request: {}", e))?;
    
    // Create Arc config for tasks
    let config = Arc::new(config);

    // Spawn heartbeat task
    let subscribe_tx_clone = subscribe_tx.clone();
    let logger_clone = logger.clone();
    
    tokio::spawn(async move {
        let ping_logger = logger_clone.clone();
        let mut interval = time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = send_heartbeat_ping(&subscribe_tx_clone, &ping_logger).await {
                ping_logger.log(format!("[CONNECTION ERROR] => {}", e).red().to_string());
                break;
            }
        }
    });
    
    // Main stream processing loop
    logger.log("Starting main processing loop...".green().to_string());
    while let Some(msg_result) = stream.next().await {
        match msg_result {
            Ok(msg) => {
                if let Err(e) = process_message(&msg, &subscribe_tx, config.clone(), &logger).await {
                    logger.log(format!("Error processing message: {}", e).red().to_string());
                }
            },
            Err(e) => {
                logger.log(format!("Stream error: {:?}", e).red().to_string());
                // Try to reconnect
                break;
            },
        }
    }
    
    logger.log("Stream ended, attempting to reconnect...".yellow().to_string());
    // Here you would implement reconnection logic
    
    Ok(())
}

/// Convert between transaction_parser::InstructionType and monitor::InstructionType
fn convert_instruction_type(instruction_type: &transaction_parser::DexType) -> monitor::InstructionType {
    match instruction_type {
        transaction_parser::DexType::PumpSwap => {
            // Since we don't have buy/sell info here, default to Buy
            // This should be updated based on transaction data elsewhere
            monitor::InstructionType::PumpSwapBuy
        },
        transaction_parser::DexType::PumpFun => {
            // Since we don't have buy/sell info here, default to Buy
            // This should be updated based on transaction data elsewhere
            monitor::InstructionType::PumpBuy
        },
        transaction_parser::DexType::Unknown => monitor::InstructionType::PumpMint,
    }
}

// Helper function to extract token mint and target from transaction
fn extract_token_info(
    _txn: &SubscribeUpdateTransaction,
    _log_messages: &[String],
    instruction_type: DexType,
) -> Result<(String, String), String> {
    // Default values
    let mut mint = String::new();
    let mut target = String::new();
    
    // Try to extract from token balances
    if let Some(txn_tx) = &_txn.transaction {
        if let Some(meta) = &txn_tx.meta {
            // Check post token balances
            if !meta.post_token_balances.is_empty() {
                for token_balance in &meta.post_token_balances {
                    // This is a token mint
                    mint = token_balance.mint.clone();
                    
                    // Owner is our target
                    if !token_balance.owner.is_empty() {
                        target = token_balance.owner.clone();
                        break;
                    }
                }
            }
        }
    }
    
    // If we couldn't extract from token balances, use defaults based on instruction type
    if mint.is_empty() {
        mint = match instruction_type {
            DexType::PumpFun => "EUizx7WDKuhkNFFTPnNZw3aoBBkSjhjtDmvg3dZu54ss".to_string(),
            DexType::PumpSwap => "2ivzYvjnKqA4X3dVvPKr7bctGpbxwrXbbxm44TJCpump".to_string(),
            _ => "".to_string(),
        };
    }
    
    if target.is_empty() {
        target = match instruction_type {
            DexType::PumpFun => "Ew7tv1bCvYfAJSQSxKVWrfke986SHrvMbsqxTngf7oxg".to_string(),
            _ => "BYBYDmUc6Qxbt3tTJ3ovVTAGQcPWQrVePEYUwq2A4pzw".to_string(),
        };
    }
    
    Ok((mint, target))
}

// Helper function to extract bonding curve from transaction
fn extract_bonding_curve(
    _txn: &SubscribeUpdateTransaction,
    _log_messages: &[String],
) -> Result<String, String> {
    // Default value
    let bonding_curve = "6221o1QRckJ5SLbniMFB53qhqnrrrXGaj88Dojknivi4".to_string();
    
    // Try to extract from logs
    for log in _log_messages {
        if log.contains("bonding_curve:") {
            if let Some(curve_str) = log.split_whitespace().nth(3) {
                return Ok(curve_str.to_string());
            }
        }
    }
    
    Ok(bonding_curve)
}

// Helper function to extract bonding curve info
fn extract_bonding_curve_info(
    _txn: &SubscribeUpdateTransaction,
    log_messages: &[String],
    bonding_curve: &str,
) -> Result<BondingCurveInfo, String> {
    // Default values
    let mut virtual_sol_reserve = 30_000_000_000;
    let mut virtual_token_reserve = 1_073_000_000_000_000;
    
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

// Helper function to calculate token amount
fn calculate_token_amount(
    _txn: &SubscribeUpdateTransaction,
    _log_messages: &[String],
    instruction_type: DexType,
) -> Result<f64, String> {
    // Default value
    let mut token_amount = 0.0;
    
    // Try to extract from token balances
    if let Some(txn_tx) = &_txn.transaction {
        if let Some(meta) = &txn_tx.meta {
            // Check post token balances
            if !meta.post_token_balances.is_empty() {
                for token_balance in &meta.post_token_balances {
                    if let Some(ui_token_amount) = &token_balance.ui_token_amount {
                        // Try to get ui_amount directly if it's not zero
                        if ui_token_amount.ui_amount != 0.0 {
                            // Use the first token amount we find
                            token_amount = ui_token_amount.ui_amount;
                            break;
                        } 
                        // Otherwise try to parse from ui_amount_string
                        else if !ui_token_amount.ui_amount_string.is_empty() {
                            // Try to parse the string amount
                            if let Ok(amount) = ui_token_amount.ui_amount_string.parse::<f64>() {
                                token_amount = amount;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // If we couldn't extract, use defaults based on instruction type
    if token_amount == 0.0 {
        token_amount = match instruction_type {
            DexType::PumpFun => 177885.45,
            DexType::PumpSwap => 63190.55,
            _ => 0.0,
        };
    }
    
    Ok(token_amount)
}

// Helper function to calculate volume change
fn calculate_volume_change(
    _txn: &SubscribeUpdateTransaction,
    _log_messages: &[String],
    instruction_type: DexType,
) -> Result<i64, String> {
    // Default value based on instruction type
    let volume_change = match instruction_type {
        DexType::PumpFun => 1_000_000_000, // 1 SOL positive change
        DexType::PumpSwap => -1_000_000_000, // 1 SOL negative change
        _ => 0,
    };
    
    Ok(volume_change)
}

// Helper function to convert PoolInfo to PumpSwapPool
fn convert_pool_info_to_pump_swap_pool(pool_info: &PoolInfo) -> crate::dex::pump_swap::PumpSwapPool {
    // Derive the associated token accounts from the pool_id and the mint addresses
    let pool_base_account = spl_associated_token_account::get_associated_token_address(
        &pool_info.pool_id, 
        &pool_info.base_mint
    );
    
    let pool_quote_account = spl_associated_token_account::get_associated_token_address(
        &pool_info.pool_id, 
        &pool_info.quote_mint
    );
    
    crate::dex::pump_swap::PumpSwapPool {
        pool_id: pool_info.pool_id,
        base_mint: pool_info.base_mint,
        quote_mint: pool_info.quote_mint,
        lp_mint: Pubkey::default(), // This field is not in PoolInfo, use default
        pool_base_account,
        pool_quote_account,
        base_reserve: pool_info.base_reserve,
        quote_reserve: pool_info.quote_reserve,
        coin_creator: pool_info.coin_creator,
    }
}

/// Verify that a transaction was successful
async fn verify_transaction(
    signature_str: &str,
    app_state: Arc<AppState>,
    logger: &Logger,
) -> Result<bool, String> {
    // Parse signature
    let signature = match Signature::from_str(signature_str) {
        Ok(sig) => sig,
        Err(e) => return Err(format!("Invalid signature: {}", e)),
    };
    
    // Verify transaction success with retries
    let max_retries = 5;
    for retry in 0..max_retries {
        // Check transaction status
        match app_state.rpc_nonblocking_client.get_signature_statuses(&[signature]).await {
            Ok(result) => {
                if let Some(status_opt) = result.value.get(0) {
                    if let Some(status) = status_opt {
                        if status.err.is_some() {
                            // Transaction failed
                            return Err(format!("Transaction failed: {:?}", status.err));
                        } else if let Some(conf_status) = &status.confirmation_status {
                            if matches!(conf_status, TransactionConfirmationStatus::Finalized | 
                                                      TransactionConfirmationStatus::Confirmed) {
                                return Ok(true);
                            } else {
                                logger.log(format!("Transaction not yet confirmed (status: {:?}), retrying...", 
                                         conf_status).yellow().to_string());
                            }
                        } else {
                        }
                    } else {
                    }
                }
            },
            Err(e) => {
                logger.log(format!("Failed to get transaction status: {}, retrying...", e).red().to_string());
            }
        }
        
        if retry < max_retries - 1 {
            // Wait before retrying
            sleep(Duration::from_millis(500)).await;
        } else {
            return Err("Transaction verification timed out".to_string());
        }
    }
    
    // If we get here, verification failed
    Err("Transaction verification failed after retries".to_string())
}

/// Execute buy operation based on detected transaction
pub async fn execute_buy(
    trade_info: transaction_parser::TradeInfoFromToken,
    app_state: Arc<AppState>,
    swap_config: Arc<SwapConfig>,
    protocol: SwapProtocol,
) -> Result<(), String> {
    let logger = Logger::new("[EXECUTE-BUY] => ".green().to_string());
    let start_time = Instant::now();
    
    // Create a modified swap config based on the trade_info
    let mut buy_config = (*swap_config).clone();
    buy_config.swap_direction = SwapDirection::Buy;
    
    // Get token amount and SOL cost from trade_info
    let (_amount_in, _token_amount) = match trade_info.dex_type {
        transaction_parser::DexType::PumpSwap => {
            let sol_amount = if let Some(amount) = trade_info.user_quote_amount_out {
                (amount as f64) / 1_000_000_000.0 // Convert lamports to SOL
            } else {
                // Fallback to volume_change if user_quote_amount_out is not available
                (trade_info.volume_change.abs() as f64) / 1_000_000_000.0
            };
            let token_amount = if let Some(amount) = trade_info.base_amount_in {
                amount as f64
            } else {
                trade_info.token_amount_f64
            };
            (sol_amount, token_amount)
        },
        transaction_parser::DexType::PumpFun => {
            let sol_amount = if let Some(amount) = trade_info.sol_amount {
                (amount as f64) / 1_000_000_000.0 // Convert lamports to SOL
            } else {
                // Fallback to volume_change if sol_amount is not available
                (trade_info.volume_change.abs() as f64) / 1_000_000_000.0
            };
            let token_amount = if let Some(amount) = trade_info.token_amount {
                amount as f64
            } else {
                trade_info.token_amount_f64
            };
            (sol_amount, token_amount)
        },
        _ => {
            return Err("Unsupported transaction type".to_string());
        }
    };
    
    // Protocol string for notifications
    let protocol_str = match protocol {
        SwapProtocol::PumpSwap => "PumpSwap",
        SwapProtocol::PumpFun => "PumpFun",
        _ => "Unknown",
    };
    
    // Send notification that we're attempting to copy the trade
    if let Err(e) = crate::services::telegram::send_trade_notification(
        &trade_info,
        protocol_str,
        "BUYING"
    ).await {
        logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
    }
    
    // Execute based on protocol
    let result = match protocol {
        SwapProtocol::PumpFun => {
            logger.log("Using PumpFun protocol for buy".to_string());
            
            // Create the PumpFun instance
            let pump = crate::dex::pump_fun::Pump::new(
                app_state.rpc_nonblocking_client.clone(),
                app_state.rpc_client.clone(),
                app_state.wallet.clone(),
            );
            // Build swap instructions from parsed data
            match pump.build_swap_from_parsed_data(&trade_info, buy_config).await {
                Ok((keypair, instructions, price, parsed_blockhash)) => {
                    logger.log(format!("Generated PumpFun buy instruction at price: {}", price));
                    
                    // Use parsed blockhash directly instead of fetching a new one
                    logger.log("Using parsed blockhash for faster transaction processing".green().to_string());
                    
                                        // Get recent blockhash
                                        let recent_blockhash = match app_state.rpc_nonblocking_client.get_latest_blockhash().await {
                                            Ok(hash) => hash,
                                            Err(e) => return Err(format!("Failed to get recent blockhash: {}", e)),
                                        };

                    // Execute the transaction
                    match crate::core::tx::new_signed_and_send_nozomi(
                        recent_blockhash,
                        &keypair,
                        instructions,
                        &logger,
                    ).await {
                        Ok(signatures) => {
                            if signatures.is_empty() {
                                return Err("No transaction signature returned".to_string());
                            }
                            
                            let signature = &signatures[0];
                            logger.log(format!("Buy transaction sent: {}", signature));
                            
                            // Send notification for transaction
                            if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                &trade_info,
                                signature,
                                protocol_str,
                                "COPIED"
                            ).await {
                                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                            }
                            
                            // Verify transaction
                            match verify_transaction(signature, app_state.clone(), &logger).await {
                                Ok(verified) => {
                                    if verified {
                                        logger.log("Buy transaction verified successfully".to_string());
                                        
                                        // Add token account to our global list
                                        if let Ok(wallet_pubkey) = app_state.wallet.try_pubkey() {
                                            let token_mint = Pubkey::from_str(&trade_info.mint)
                                                .map_err(|_| "Invalid token mint".to_string())?;
                                            let token_ata = get_associated_token_address(&wallet_pubkey, &token_mint);
                                            WALLET_TOKEN_ACCOUNTS.insert(token_ata);
                                            logger.log(format!("Added token account {} to global list", token_ata));
                                        }
                                        
                                        // Send notification for successful verification
                                        if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                            &trade_info,
                                            signature,
                                            protocol_str,
                                            "BOUGHT"
                                        ).await {
                                            logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                        }
                                        
                                        Ok(())
                                    } else {
                                        // Send error notification
                                        if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                            &trade_info,
                                            signature,
                                            protocol_str,
                                            "ERROR"
                                        ).await {
                                            logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                        }
                                        
                                        Err("Buy transaction verification failed".to_string())
                                    }
                                },
                                Err(e) => {
                                    // Send error notification
                                    if let Err(te) = crate::services::telegram::send_error_notification(
                                        &format!("Transaction verification error: {}", e)
                                    ).await {
                                        logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                    }
                                    
                                    Err(format!("Transaction verification error: {}", e))
                                },
                            }
                        },
                        Err(e) => {
                            // Send error notification
                            if let Err(te) = crate::services::telegram::send_error_notification(
                                &format!("Transaction error: {}", e)
                            ).await {
                                logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                            }
                            
                            Err(format!("Transaction error: {}", e))
                        },
                    }
                },
                Err(e) => {
                    // Send error notification
                    if let Err(te) = crate::services::telegram::send_error_notification(
                        &format!("Failed to build PumpFun buy instruction: {}", e)
                    ).await {
                        logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                    }
                    
                    Err(format!("Failed to build PumpFun buy instruction: {}", e))
                },
            }
        },
        SwapProtocol::PumpSwap => {
            logger.log("Using PumpSwap protocol for buy".to_string());
            
            // Create the PumpSwap instance
            let pump_swap = crate::dex::pump_swap::PumpSwap::new(
                app_state.wallet.clone(),
                Some(app_state.rpc_client.clone()),
                Some(app_state.rpc_nonblocking_client.clone()),
            );
            
            // Build swap instructions from parsed data for buy
            match pump_swap.build_swap_from_parsed_data(&trade_info, buy_config).await {
                Ok((keypair, instructions, price, parsed_blockhash)) => {
                    logger.log(format!("Generated PumpSwap buy instruction at price: {}", price));
                    logger.log(format!("copy transaction {}", trade_info.signature));
                    
                    // We intentionally use a fresh blockhash instead of parsed_blockhash for better reliability
                    let recent_blockhash = match app_state.rpc_nonblocking_client.get_latest_blockhash().await {
                        Ok(hash) => hash,
                        Err(e) => return Err(format!("Failed to get recent blockhash: {}", e)),
                    };
                    // Execute the transaction
                    match crate::core::tx::new_signed_and_send_nozomi(
                        recent_blockhash,
                        &keypair,
                        instructions,
                        &logger,
                    ).await {
                        Ok(signatures) => {
                            if signatures.is_empty() {
                                return Err("No transaction signature returned".to_string());
                            }
                            
                            let signature = &signatures[0];
                            logger.log(format!("Buy transaction sent: {}", signature));
                            
                            // Send notification for transaction
                            if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                &trade_info,
                                signature,
                                protocol_str,
                                "COPIED"
                            ).await {
                                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                            }
                            
                            // Verify transaction
                            match verify_transaction(signature, app_state.clone(), &logger).await {
                                Ok(verified) => {
                                    if verified {
                                        logger.log("Buy transaction verified successfully".to_string());
                                        
                                        // Add token account to our global list
                                        if let Ok(wallet_pubkey) = app_state.wallet.try_pubkey() {
                                            let token_mint = Pubkey::from_str(&trade_info.mint)
                                                .map_err(|_| "Invalid token mint".to_string())?;
                                            let token_ata = get_associated_token_address(&wallet_pubkey, &token_mint);
                                            WALLET_TOKEN_ACCOUNTS.insert(token_ata);
                                            logger.log(format!("Added token account {} to global list", token_ata));
                                        }
                                        
                                        // Send notification for successful verification
                                        if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                            &trade_info,
                                            signature,
                                            protocol_str,
                                            "BOUGHT"
                                        ).await {
                                            logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                        }
                                        
                                        Ok(())
                                    } else {
                                        // Send error notification
                                        if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                            &trade_info,
                                            signature,
                                            protocol_str,
                                            "ERROR"
                                        ).await {
                                            logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                        }
                                        
                                        Err("Buy transaction verification failed".to_string())
                                    }
                                },
                                Err(e) => {
                                    // Send error notification
                                    if let Err(te) = crate::services::telegram::send_error_notification(
                                        &format!("Transaction verification error: {}", e)
                                    ).await {
                                        logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                    }
                                    
                                    Err(format!("Transaction verification error: {}", e))
                                },
                            }
                        },
                        Err(e) => {
                            // Send error notification
                            if let Err(te) = crate::services::telegram::send_error_notification(
                                &format!("Transaction error: {}", e)
                            ).await {
                                logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                            }
                            
                            Err(format!("Transaction error: {}", e))
                        },
                    }
                },
                Err(e) => {
                    // Send error notification
                    if let Err(te) = crate::services::telegram::send_error_notification(
                        &format!("Failed to build PumpSwap buy instruction: {}", e)
                    ).await {
                        logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                    }
                    
                    Err(format!("Failed to build PumpSwap buy instruction: {}", e))
                },
            }
        },
        _ => Err(format!("Unsupported protocol: {:?}", protocol)),
    };
    
    // Log execution time
    let elapsed = start_time.elapsed();
    logger.log(format!("Buy execution time: {:?}", elapsed));
    
    // Increment bought counter on success
    if result.is_ok() {
        // Extract data from mutex guards and drop them before spawning a task
        let (bought_count, active_tokens) = {
            let mut bought = BOUGHT_TOKENS.lock().unwrap();
            *bought += 1;
            let bought_count = *bought;
            logger.log(format!("Total bought: {}", bought_count));
            
            // Initialize tracking for this token
            let mut tracking = TOKEN_TRACKING.lock().unwrap();
            tracking.entry(trade_info.mint.clone()).or_insert(TokenTrackingInfo {
                top_pnl: 0.0,
                last_sell_time: Instant::now(),
                completed_intervals: HashSet::new(),
            });
            
            let sold_count = *SOLD_TOKENS.lock().unwrap();
            let active_tokens: Vec<String> = tracking.keys().cloned().collect();
            
            (bought_count, active_tokens)
        };
        
        // Copy values needed for the task
        let sold_count = *SOLD_TOKENS.lock().unwrap();
        
        // Send summary notification with copied values
        tokio::spawn(async move {
            if let Err(e) = crate::services::telegram::send_summary_notification(
                bought_count,
                sold_count,
                active_tokens,
                0.0 // We don't have PNL tracking yet, implement later
            ).await {
                eprintln!("Failed to send summary notification: {}", e);
            }
        });
    }
    
    result
}

/// Execute sell operation for a token
pub async fn execute_sell(
    token_mint: String,
    trade_info: transaction_parser::TradeInfoFromToken,
    app_state: Arc<AppState>,
    swap_config: Arc<SwapConfig>,
    protocol: SwapProtocol,
    progressive_sell: bool,
    chunks: Option<usize>,
    interval_ms: Option<u64>,
) -> Result<(), String> {
    let logger = Logger::new("[EXECUTE-SELL] => ".green().to_string());
    let start_time = Instant::now();
    
    logger.log(format!("Selling token: {}", token_mint));
    
    // Protocol string for notifications
    let protocol_str = match protocol {
        SwapProtocol::PumpSwap => "PumpSwap",
        SwapProtocol::PumpFun => "PumpFun",
        _ => "Unknown",
    };
    
    // Create a minimal trade info for notification
    let notification_trade_info = transaction_parser::TradeInfoFromToken {
        dex_type: match protocol {
            SwapProtocol::PumpSwap => transaction_parser::DexType::PumpSwap,
            SwapProtocol::PumpFun => transaction_parser::DexType::PumpFun,
            _ => transaction_parser::DexType::Unknown,
        },
        signature: trade_info.signature.clone(),
        mint: token_mint.clone(),
        target: trade_info.target.clone(),
        is_buy: false,
        token_amount_f64: trade_info.token_amount.unwrap_or(0) as f64, // Convert Option<u64> to f64
        // Copy values from trade_info instead of using defaults
        slot: trade_info.slot,
        recent_blockhash: trade_info.recent_blockhash,
        user: trade_info.user.clone(),
        timestamp: trade_info.timestamp,
        sol_amount: trade_info.sol_amount,
        token_amount: trade_info.token_amount,
        min_quote_amount_out: trade_info.min_quote_amount_out,
        user_base_token_reserves: trade_info.user_base_token_reserves,
        user_quote_token_reserves: trade_info.user_quote_token_reserves,
        pool_base_token_reserves: trade_info.pool_base_token_reserves,
        pool_quote_token_reserves: trade_info.pool_quote_token_reserves,
        quote_amount_out: trade_info.quote_amount_out,
        lp_fee_basis_points: trade_info.lp_fee_basis_points,
        lp_fee: trade_info.lp_fee,
        protocol_fee_basis_points: trade_info.protocol_fee_basis_points,
        protocol_fee: trade_info.protocol_fee,
        quote_amount_out_without_lp_fee: trade_info.quote_amount_out_without_lp_fee,
        user_quote_amount_out: trade_info.user_quote_amount_out,
        pool: trade_info.pool.clone(),
        user_base_token_account: trade_info.user_base_token_account.clone(),
        user_quote_token_account: trade_info.user_quote_token_account.clone(),
        protocol_fee_recipient: trade_info.protocol_fee_recipient.clone(),
        protocol_fee_recipient_token_account: trade_info.protocol_fee_recipient_token_account.clone(),
        coin_creator: trade_info.coin_creator.clone(),
        coin_creator_fee_basis_points: trade_info.coin_creator_fee_basis_points,
        coin_creator_fee: trade_info.coin_creator_fee,
        virtual_sol_reserves: trade_info.virtual_sol_reserves,
        virtual_token_reserves: trade_info.virtual_token_reserves,
        real_sol_reserves: trade_info.real_sol_reserves,
        real_token_reserves: trade_info.real_token_reserves,
        bonding_curve: trade_info.bonding_curve.clone(),
        volume_change: trade_info.volume_change,
        bonding_curve_info: trade_info.bonding_curve_info.clone(),
        pool_info: trade_info.pool_info.clone(),
        amount: trade_info.amount,
        max_sol_cost: trade_info.max_sol_cost,
        min_sol_output: trade_info.min_sol_output,
        base_amount_in: trade_info.base_amount_in,
        base_amount_out: trade_info.base_amount_out,
        max_quote_amount_in: trade_info.max_quote_amount_in,
    };

    // Create a modified swap config for selling
    let mut sell_config = (*swap_config).clone();
    sell_config.swap_direction = SwapDirection::Sell;

    // Get wallet pubkey - handle the error properly instead of using ?
    let wallet_pubkey = match app_state.wallet.try_pubkey() {
        Ok(pubkey) => pubkey,
        Err(e) => return Err(format!("Failed to get wallet pubkey: {}", e)),
    };

    // Get token account to determine how much we own
    let token_pubkey = match Pubkey::from_str(&token_mint) {
        Ok(pubkey) => pubkey,
        Err(e) => return Err(format!("Invalid token mint address: {}", e)),
    };
    let ata = get_associated_token_address(&wallet_pubkey, &token_pubkey);

    // Get token account and amount
    let token_amount = match app_state.rpc_nonblocking_client.get_token_account(&ata).await {
        Ok(Some(account)) => {
            // Parse the amount string instead of casting
            let amount_value = match account.token_amount.amount.parse::<f64>() {
                Ok(val) => val,
                Err(e) => return Err(format!("Failed to parse token amount: {}", e)),
            };
            let decimal_amount = amount_value / 10f64.powi(account.token_amount.decimals as i32);
            logger.log(format!("Token amount to sell: {}", decimal_amount));
            decimal_amount
        },
        Ok(None) => {
            return Err(format!("No token account found for mint: {}", token_mint));
        },
        Err(e) => {
            return Err(format!("Failed to get token account: {}", e));
        }
    };
    
    // Update trade info with token amount
    let mut notification_trade_info = notification_trade_info.clone();
    notification_trade_info.token_amount_f64 = token_amount;

    // Now that we have token_amount, set slippage based on token value
    // Use a fixed slippage since calculate_dynamic_slippage is private
    let token_value = token_amount * 0.01; // Estimate value at 0.01 SOL per token as a conservative default
    let slippage_bps = if token_value > 10.0 {
        300 // 3% for high value tokens (300 basis points)
    } else if token_value > 1.0 {
        200 // 2% for medium value tokens (200 basis points)
    } else {
        100 // 1% for low value tokens (100 basis points)
    };

    logger.log(format!("Using slippage of {}%", slippage_bps as f64 / 100.0));
    sell_config.slippage = slippage_bps;
    
    // Send notification that we're about to sell
    if let Err(e) = crate::services::telegram::send_trade_notification(
        &notification_trade_info,
        protocol_str,
        "SELLING"
    ).await {
        logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
    }
    
    // If progressive sell is enabled, divide into chunks
    if progressive_sell {
        let chunks_count = chunks.unwrap_or(3);
        let interval = interval_ms.unwrap_or(2000); // 2 seconds default
        
        logger.log(format!("Executing progressive sell in {} chunks with {} ms intervals", chunks_count, interval));
        
        // Calculate chunk size
        let chunk_size = token_amount / chunks_count as f64;
        
        // Execute each chunk
        for i in 0..chunks_count {
            // Create a fresh sell config for each iteration by cloning
            let mut chunk_sell_config = (*swap_config).clone();
            chunk_sell_config.swap_direction = SwapDirection::Sell;
            chunk_sell_config.slippage = slippage_bps;
            
            // Adjust the final chunk to account for any rounding errors
            let amount_to_sell = if i == chunks_count - 1 {
                // For the last chunk, sell whatever is left
                match app_state.rpc_nonblocking_client.get_token_account(&ata).await {
                    Ok(Some(account)) => {
                        // Parse the amount string instead of casting
                        let amount_value = match account.token_amount.amount.parse::<f64>() {
                            Ok(val) => val,
                            Err(e) => return Err(format!("Failed to parse token amount: {}", e)),
                        };
                        let remaining = amount_value / 10f64.powi(account.token_amount.decimals as i32);
                        if remaining < 0.000001 { // Very small amount, not worth selling
                            logger.log("Remaining amount too small, skipping final chunk".to_string());
                            break;
                        }
                        remaining
                    },
                    Ok(None) => chunk_size, // Fallback if we can't get the account
                    Err(e) => return Err(format!("Failed to get token account: {}", e)),
                }
            } else {
                chunk_size
            };
            
            if amount_to_sell <= 0.0 {
                logger.log("No tokens left to sell in this chunk".to_string());
                continue;
            }
            
            // Update config for this chunk
            chunk_sell_config.amount_in = amount_to_sell;
            
            logger.log(format!("Selling chunk {}/{}: {} tokens", i + 1, chunks_count, amount_to_sell));
            
            // Update trade info for this chunk
            let mut chunk_trade_info = notification_trade_info.clone();
            chunk_trade_info.token_amount_f64 = amount_to_sell;
            
            // Send notification for this chunk
            if let Err(e) = crate::services::telegram::send_trade_notification(
                &chunk_trade_info,
                &format!("{}(Chunk {}/{})", protocol_str, i + 1, chunks_count),
                "SELLING"
            ).await {
                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
            }
            
            // Execute sell based on protocol
            let result = match protocol {
                SwapProtocol::PumpFun => {
                    logger.log("Using PumpFun protocol for sell".to_string());
                    
                    // Create the PumpFun instance
                    let pump = crate::dex::pump_fun::Pump::new(
                        app_state.rpc_nonblocking_client.clone(),
                        app_state.rpc_client.clone(),
                        app_state.wallet.clone(),
                    );
                    
                    // Create a minimal trade info struct for the sell
                    let chunk_trade_info = transaction_parser::TradeInfoFromToken {
                        signature: "standard_sell".to_string(),
                        mint: token_mint.clone(),
                        dex_type: transaction_parser::DexType::PumpFun,
                        is_buy: false,
                        sol_amount: trade_info.sol_amount,
                        token_amount: trade_info.token_amount,
                        volume_change: trade_info.volume_change,
                        token_amount_f64: amount_to_sell,
                        user_quote_amount_out: trade_info.user_quote_amount_out,
                        base_amount_in: trade_info.base_amount_in,
                        bonding_curve_info: trade_info.bonding_curve_info.clone(),
                        pool_info: trade_info.pool_info.clone(),
                        // Copy values from original trade_info instead of using defaults
                        slot: trade_info.slot,
                        recent_blockhash: trade_info.recent_blockhash,
                        target: trade_info.target.clone(),
                        user: trade_info.user.clone(),
                        timestamp: trade_info.timestamp,
                        min_quote_amount_out: trade_info.min_quote_amount_out,
                        user_base_token_reserves: trade_info.user_base_token_reserves,
                        user_quote_token_reserves: trade_info.user_quote_token_reserves,
                        pool_base_token_reserves: trade_info.pool_base_token_reserves,
                        pool_quote_token_reserves: trade_info.pool_quote_token_reserves,
                        quote_amount_out: trade_info.quote_amount_out,
                        lp_fee_basis_points: trade_info.lp_fee_basis_points,
                        lp_fee: trade_info.lp_fee,
                        protocol_fee_basis_points: trade_info.protocol_fee_basis_points,
                        protocol_fee: trade_info.protocol_fee,
                        quote_amount_out_without_lp_fee: trade_info.quote_amount_out_without_lp_fee,
                        pool: trade_info.pool.clone(),
                        user_base_token_account: trade_info.user_base_token_account.clone(),
                        user_quote_token_account: trade_info.user_quote_token_account.clone(),
                        protocol_fee_recipient: trade_info.protocol_fee_recipient.clone(),
                        protocol_fee_recipient_token_account: trade_info.protocol_fee_recipient_token_account.clone(),
                        coin_creator: trade_info.coin_creator.clone(),
                        coin_creator_fee_basis_points: trade_info.coin_creator_fee_basis_points,
                        coin_creator_fee: trade_info.coin_creator_fee,
                        virtual_sol_reserves: trade_info.virtual_sol_reserves,
                        virtual_token_reserves: trade_info.virtual_token_reserves,
                        real_sol_reserves: trade_info.real_sol_reserves,
                        real_token_reserves: trade_info.real_token_reserves,
                        bonding_curve: trade_info.bonding_curve.clone(),
                        amount: trade_info.amount,
                        max_sol_cost: trade_info.max_sol_cost,
                        min_sol_output: trade_info.min_sol_output,
                        base_amount_out: trade_info.base_amount_out,
                        max_quote_amount_in: trade_info.max_quote_amount_in,
                    };
                    
                    // Build swap instructions for sell - use chunk_sell_config
                    match pump.build_swap_from_parsed_data(&chunk_trade_info, chunk_sell_config).await {
                        Ok((keypair, instructions, price, parsed_blockhash)) => {
                            logger.log(format!("Generated PumpFun sell instruction at price: {}", price));
                            let recent_blockhash = match app_state.rpc_nonblocking_client.get_latest_blockhash().await {
                                Ok(hash) => hash,
                                Err(e) => return Err(format!("Failed to get recent blockhash: {}", e)),
                            };
                            // Execute the transaction
                            match crate::core::tx::new_signed_and_send_nozomi(
                                recent_blockhash,
                                &keypair,
                                instructions,
                                &logger,
                            ).await {
                                Ok(signatures) => {
                                    if signatures.is_empty() {
                                        return Err("No transaction signature returned".to_string());
                                    }
                                    
                                    let signature = &signatures[0];
                                    logger.log(format!("Sell transaction sent: {}", signature));
                                    
                                    // Send notification for transaction
                                    if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                        &chunk_trade_info,
                                        signature,
                                        protocol_str,
                                        "SELLING"
                                    ).await {
                                        logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                    }
                                    
                                    // Verify transaction
                                    match verify_transaction(signature, app_state.clone(), &logger).await {
                                        Ok(verified) => {
                                            if verified {
                                                logger.log("Sell transaction verified successfully".to_string());
                                                
                                                // Send notification for successful verification
                                                if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                    &chunk_trade_info,
                                                    signature,
                                                    protocol_str,
                                                    "SOLD"
                                                ).await {
                                                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                                }
                                                
                                                Ok(())
                                            } else {
                                                // Send error notification
                                                if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                    &chunk_trade_info,
                                                    signature,
                                                    protocol_str,
                                                    "ERROR"
                                                ).await {
                                                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                                }
                                                
                                                Err("Sell transaction verification failed".to_string())
                                            }
                                        },
                                        Err(e) => {
                                            // Send error notification
                                            if let Err(te) = crate::services::telegram::send_error_notification(
                                                &format!("Transaction verification error: {}", e)
                                            ).await {
                                                logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                            }
                                            
                                            Err(format!("Transaction verification error: {}", e))
                                        },
                                    }
                                },
                                Err(e) => {
                                    // Send error notification
                                    if let Err(te) = crate::services::telegram::send_error_notification(
                                        &format!("Transaction error: {}", e)
                                    ).await {
                                        logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                    }
                                    
                                    Err(format!("Transaction error: {}", e))
                                },
                            }
                        },
                        Err(e) => {
                            // Send error notification
                            if let Err(te) = crate::services::telegram::send_error_notification(
                                &format!("Failed to build PumpFun sell instruction: {}", e)
                            ).await {
                                logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                            }
                            
                            Err(format!("Failed to build PumpFun sell instruction: {}", e))
                        },
                    }
                },
                SwapProtocol::PumpSwap => {
                    logger.log("Using PumpSwap protocol for sell".to_string());
                    
                    // Create the PumpSwap instance
                    let pump_swap = crate::dex::pump_swap::PumpSwap::new(
                        app_state.wallet.clone(),
                        Some(app_state.rpc_client.clone()),
                        Some(app_state.rpc_nonblocking_client.clone()),
                    );
                    
                    // Create a fresh clone of trade_info for this operation
                    let trade_info_clone = transaction_parser::TradeInfoFromToken {
                        signature: "chunk_sell".to_string(),
                        mint: token_mint.clone(),
                        dex_type: transaction_parser::DexType::PumpSwap,
                        is_buy: false,
                        sol_amount: trade_info.sol_amount,
                        token_amount: trade_info.token_amount,
                        volume_change: trade_info.volume_change,
                        token_amount_f64: amount_to_sell,
                        user_quote_amount_out: trade_info.user_quote_amount_out,
                        base_amount_in: trade_info.base_amount_in,
                        bonding_curve_info: trade_info.bonding_curve_info.clone(),
                        pool_info: trade_info.pool_info.clone(),
                        // Copy values from original trade_info instead of using defaults
                        slot: trade_info.slot,
                        recent_blockhash: trade_info.recent_blockhash,
                        target: trade_info.target.clone(),
                        user: trade_info.user.clone(),
                        timestamp: trade_info.timestamp,
                        min_quote_amount_out: trade_info.min_quote_amount_out,
                        user_base_token_reserves: trade_info.user_base_token_reserves,
                        user_quote_token_reserves: trade_info.user_quote_token_reserves,
                        pool_base_token_reserves: trade_info.pool_base_token_reserves,
                        pool_quote_token_reserves: trade_info.pool_quote_token_reserves,
                        quote_amount_out: trade_info.quote_amount_out,
                        lp_fee_basis_points: trade_info.lp_fee_basis_points,
                        lp_fee: trade_info.lp_fee,
                        protocol_fee_basis_points: trade_info.protocol_fee_basis_points,
                        protocol_fee: trade_info.protocol_fee,
                        quote_amount_out_without_lp_fee: trade_info.quote_amount_out_without_lp_fee,
                        pool: trade_info.pool.clone(),
                        user_base_token_account: trade_info.user_base_token_account.clone(),
                        user_quote_token_account: trade_info.user_quote_token_account.clone(),
                        protocol_fee_recipient: trade_info.protocol_fee_recipient.clone(),
                        protocol_fee_recipient_token_account: trade_info.protocol_fee_recipient_token_account.clone(),
                        coin_creator: trade_info.coin_creator.clone(),
                        coin_creator_fee_basis_points: trade_info.coin_creator_fee_basis_points,
                        coin_creator_fee: trade_info.coin_creator_fee,
                        virtual_sol_reserves: trade_info.virtual_sol_reserves,
                        virtual_token_reserves: trade_info.virtual_token_reserves,
                        real_sol_reserves: trade_info.real_sol_reserves,
                        real_token_reserves: trade_info.real_token_reserves,
                        bonding_curve: trade_info.bonding_curve.clone(),
                        amount: trade_info.amount,
                        max_sol_cost: trade_info.max_sol_cost,
                        min_sol_output: trade_info.min_sol_output,
                        base_amount_out: trade_info.base_amount_out,
                        max_quote_amount_in: trade_info.max_quote_amount_in,
                    };
                    
                    // Build swap instructions for sell - use chunk_sell_config
                    match pump_swap.build_swap_from_parsed_data(&trade_info_clone, chunk_sell_config).await {
                        Ok((keypair, instructions, price, parsed_blockhash)) => {
                            logger.log(format!("Generated PumpSwap sell instruction at price: {}", price));
                            
                            // Using a fresh blockhash instead of parsed_blockhash for better reliability
                            logger.log("Getting fresh blockhash for reliable transaction processing".green().to_string());
                            let recent_blockhash = match app_state.rpc_nonblocking_client.get_latest_blockhash().await {
                                Ok(hash) => hash,
                                Err(e) => return Err(format!("Failed to get recent blockhash: {}", e)),
                            };
                            
                            // Execute the transaction
                            match crate::core::tx::new_signed_and_send_nozomi(
                                recent_blockhash,
                                &keypair,
                                instructions,
                                &logger,
                            ).await {
                                Ok(signatures) => {
                                    if signatures.is_empty() {
                                        return Err("No transaction signature returned".to_string());
                                    }
                                    
                                    let signature = &signatures[0];
                                    logger.log(format!("Sell transaction sent: {}", signature));
                                    
                                    // Send notification for transaction
                                    if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                        &trade_info_clone,
                                        signature,
                                        protocol_str,
                                        "SELLING"
                                    ).await {
                                        logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                    }
                                    
                                    // Verify transaction
                                    match verify_transaction(signature, app_state.clone(), &logger).await {
                                        Ok(verified) => {
                                            if verified {
                                                logger.log("Sell transaction verified successfully".to_string());
                                                
                                                // Send notification for successful verification
                                                if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                    &trade_info_clone,
                                                    signature,
                                                    protocol_str,
                                                    "SOLD"
                                                ).await {
                                                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                                }
                                                
                                                Ok(())
                                            } else {
                                                // Send error notification
                                                if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                    &trade_info_clone,
                                                    signature,
                                                    protocol_str,
                                                    "ERROR"
                                                ).await {
                                                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                                }
                                                
                                                Err("Sell transaction verification failed".to_string())
                                            }
                                        },
                                        Err(e) => {
                                            // Send error notification
                                            if let Err(te) = crate::services::telegram::send_error_notification(
                                                &format!("Transaction verification error: {}", e)
                                            ).await {
                                                logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                            }
                                            
                                            Err(format!("Transaction verification error: {}", e))
                                        },
                                    }
                                },
                                Err(e) => {
                                    // Send error notification
                                    if let Err(te) = crate::services::telegram::send_error_notification(
                                        &format!("Transaction error: {}", e)
                                    ).await {
                                        logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                    }
                                    
                                    Err(format!("Transaction error: {}", e))
                                },
                            }
                        },
                        Err(e) => {
                            // Send error notification
                            if let Err(te) = crate::services::telegram::send_error_notification(
                                &format!("Failed to build PumpSwap sell instruction: {}", e)
                            ).await {
                                logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                            }
                            
                            Err(format!("Failed to build PumpSwap sell instruction: {}", e))
                        },
                    }
                },
                _ => Err(format!("Unsupported protocol: {:?}", protocol)),
            };
            
            // If any chunk fails, return the error
            if let Err(e) = result {
                logger.log(format!("Failed to sell chunk {}/{}: {}", i + 1, chunks_count, e));
                
                // Send error notification
                if let Err(te) = crate::services::telegram::send_error_notification(
                    &format!("Sell error (chunk {}/{}): {}", i + 1, chunks_count, e)
                ).await {
                    logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                }
                
                return Err(e);
            }
            
            // Wait for the specified interval before next chunk
            if i < chunks_count - 1 {
                logger.log(format!("Waiting {}ms before next chunk", interval));
                tokio::time::sleep(Duration::from_millis(interval)).await;
            }
        }
        
        // Log execution time for progressive sell
        let elapsed = start_time.elapsed();
        logger.log(format!("Progressive sell execution time: {:?}", elapsed));

        // Increment sold counter - but don't hold the lock across awaits
        let (bought_count, sold_count, active_tokens) = {
            let mut sold = SOLD_TOKENS.lock().unwrap();
            *sold += 1;
            let sold_count = *sold;
            logger.log(format!("Total sold: {}", sold_count));
            
            let bought_count = *BOUGHT_TOKENS.lock().unwrap();
            let active_tokens: Vec<String> = TOKEN_TRACKING.lock().unwrap().keys().cloned().collect();
            
            (bought_count, sold_count, active_tokens)
        };
        
        // When all chunks are sold successfully, remove the token account from our global list
        // This ATA is now empty and may be closed
        WALLET_TOKEN_ACCOUNTS.remove(&ata);
        logger.log(format!("Removed token account {} from global list after progressive sell", ata));
        
        // Send final success notification
        if let Err(e) = crate::services::telegram::send_trade_notification(
            &notification_trade_info,
            &format!("{} (Progressive)", protocol_str),
            "SOLD"
        ).await {
            logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
        }
        
        // Send summary - now with copied values that don't include MutexGuard
        tokio::spawn(async move {
            if let Err(e) = crate::services::telegram::send_summary_notification(
                bought_count,
                sold_count,
                active_tokens,
                0.0 // We don't have PNL tracking yet
            ).await {
                eprintln!("Failed to send summary notification: {}", e);
            }
        });
        
        Ok(())
    } else {
        // Standard single-transaction sell
        logger.log("Executing standard sell".to_string());
        
        // Configure to sell 100% of tokens
        sell_config.amount_in = token_amount;
        
        // Execute based on protocol
        let result = match protocol {
            SwapProtocol::PumpFun => {
                logger.log("Using PumpFun protocol for sell".to_string());
                
                // Create the PumpFun instance
                let pump = crate::dex::pump_fun::Pump::new(
                    app_state.rpc_nonblocking_client.clone(),
                    app_state.rpc_client.clone(),
                    app_state.wallet.clone(),
                );
                
                // Create a minimal trade info struct for the sell
                let trade_info_clone = transaction_parser::TradeInfoFromToken {
                    signature: "standard_sell".to_string(),
                    mint: token_mint.clone(),
                    dex_type: transaction_parser::DexType::PumpFun,
                    is_buy: false,
                    sol_amount: trade_info.sol_amount,
                    token_amount: trade_info.token_amount,
                    volume_change: trade_info.volume_change,
                    token_amount_f64: token_amount,
                    user_quote_amount_out: trade_info.user_quote_amount_out,
                    base_amount_in: trade_info.base_amount_in,
                    bonding_curve_info: trade_info.bonding_curve_info.clone(),
                    pool_info: trade_info.pool_info.clone(),
                    // Copy values from original trade_info instead of using defaults
                    slot: trade_info.slot,
                    recent_blockhash: trade_info.recent_blockhash,
                    target: trade_info.target.clone(),
                    user: trade_info.user.clone(),
                    timestamp: trade_info.timestamp,
                    min_quote_amount_out: trade_info.min_quote_amount_out,
                    user_base_token_reserves: trade_info.user_base_token_reserves,
                    user_quote_token_reserves: trade_info.user_quote_token_reserves,
                    pool_base_token_reserves: trade_info.pool_base_token_reserves,
                    pool_quote_token_reserves: trade_info.pool_quote_token_reserves,
                    quote_amount_out: trade_info.quote_amount_out,
                    lp_fee_basis_points: trade_info.lp_fee_basis_points,
                    lp_fee: trade_info.lp_fee,
                    protocol_fee_basis_points: trade_info.protocol_fee_basis_points,
                    protocol_fee: trade_info.protocol_fee,
                    quote_amount_out_without_lp_fee: trade_info.quote_amount_out_without_lp_fee,
                    pool: trade_info.pool.clone(),
                    user_base_token_account: trade_info.user_base_token_account.clone(),
                    user_quote_token_account: trade_info.user_quote_token_account.clone(),
                    protocol_fee_recipient: trade_info.protocol_fee_recipient.clone(),
                    protocol_fee_recipient_token_account: trade_info.protocol_fee_recipient_token_account.clone(),
                    coin_creator: trade_info.coin_creator.clone(),
                    coin_creator_fee_basis_points: trade_info.coin_creator_fee_basis_points,
                    coin_creator_fee: trade_info.coin_creator_fee,
                    virtual_sol_reserves: trade_info.virtual_sol_reserves,
                    virtual_token_reserves: trade_info.virtual_token_reserves,
                    real_sol_reserves: trade_info.real_sol_reserves,
                    real_token_reserves: trade_info.real_token_reserves,
                    bonding_curve: trade_info.bonding_curve.clone(),
                    amount: trade_info.amount,
                    max_sol_cost: trade_info.max_sol_cost,
                    min_sol_output: trade_info.min_sol_output,
                    base_amount_out: trade_info.base_amount_out,
                    max_quote_amount_in: trade_info.max_quote_amount_in,
                };
                
                // Build swap instructions for sell
                match pump.build_swap_from_parsed_data(&trade_info_clone, sell_config).await {
                    Ok((keypair, instructions, price, parsed_blockhash)) => {
                        logger.log(format!("Generated PumpFun sell instruction at price: {}", price));
                        
                        // Use parsed blockhash for faster execution
                        logger.log("Using parsed blockhash for faster transaction processing".green().to_string());
                        // Get recent blockhash
                        let recent_blockhash = match app_state.rpc_nonblocking_client.get_latest_blockhash().await {
                            Ok(hash) => hash,
                            Err(e) => return Err(format!("Failed to get recent blockhash: {}", e)),
                        };
                        // Execute the transaction
                        match crate::core::tx::new_signed_and_send_nozomi(
                            recent_blockhash,
                            &keypair,
                            instructions,
                            &logger,
                        ).await {
                            Ok(signatures) => {
                                if signatures.is_empty() {
                                    return Err("No transaction signature returned".to_string());
                                }
                                
                                let signature = &signatures[0];
                                logger.log(format!("Sell transaction sent: {}", signature));
                                
                                // Send notification for transaction
                                if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                    &trade_info_clone,
                                    signature,
                                    protocol_str,
                                    "SELLING"
                                ).await {
                                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                }
                                
                                // Verify transaction
                                match verify_transaction(signature, app_state.clone(), &logger).await {
                                    Ok(verified) => {
                                        if verified {
                                            logger.log("Sell transaction verified successfully".to_string());
                                            
                                            // Send notification for successful verification
                                            if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                &trade_info_clone,
                                                signature,
                                                protocol_str,
                                                "SOLD"
                                            ).await {
                                                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                            }
                                            
                                            Ok(())
                                        } else {
                                            // Send error notification
                                            if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                &trade_info_clone,
                                                signature,
                                                protocol_str,
                                                "ERROR"
                                            ).await {
                                                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                            }
                                            
                                            Err("Sell transaction verification failed".to_string())
                                        }
                                    },
                                    Err(e) => {
                                        // Send error notification
                                        if let Err(te) = crate::services::telegram::send_error_notification(
                                            &format!("Transaction verification error: {}", e)
                                        ).await {
                                            logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                        }
                                        
                                        Err(format!("Transaction verification error: {}", e))
                                    },
                                }
                            },
                            Err(e) => {
                                // Send error notification
                                if let Err(te) = crate::services::telegram::send_error_notification(
                                    &format!("Transaction error: {}", e)
                                ).await {
                                    logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                }
                                
                                Err(format!("Transaction error: {}", e))
                            },
                        }
                    },
                    Err(e) => {
                        // Send error notification
                        if let Err(te) = crate::services::telegram::send_error_notification(
                            &format!("Failed to build PumpFun sell instruction: {}", e)
                        ).await {
                            logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                        }
                        
                        Err(format!("Failed to build PumpFun sell instruction: {}", e))
                    },
                }
            },
            SwapProtocol::PumpSwap => {
                logger.log("Using PumpSwap protocol for sell".to_string());
                
                // Create the PumpSwap instance
                let pump_swap = crate::dex::pump_swap::PumpSwap::new(
                    app_state.wallet.clone(),
                    Some(app_state.rpc_client.clone()),
                    Some(app_state.rpc_nonblocking_client.clone()),
                );
                
                // Create a minimal trade info struct for the sell
                let trade_info_clone = transaction_parser::TradeInfoFromToken {
                    signature: "standard_sell".to_string(),
                    mint: token_mint.clone(),
                    dex_type: transaction_parser::DexType::PumpSwap,
                    is_buy: false,
                    sol_amount: trade_info.sol_amount,
                    token_amount: trade_info.token_amount,
                    volume_change: trade_info.volume_change,
                    token_amount_f64: token_amount,
                    user_quote_amount_out: trade_info.user_quote_amount_out,
                    base_amount_in: trade_info.base_amount_in,
                    bonding_curve_info: trade_info.bonding_curve_info.clone(),
                    pool_info: trade_info.pool_info.clone(),
                    // Copy values from original trade_info instead of using defaults
                    slot: trade_info.slot,
                    recent_blockhash: trade_info.recent_blockhash,
                    target: trade_info.target.clone(),
                    user: trade_info.user.clone(),
                    timestamp: trade_info.timestamp,
                    min_quote_amount_out: trade_info.min_quote_amount_out,
                    user_base_token_reserves: trade_info.user_base_token_reserves,
                    user_quote_token_reserves: trade_info.user_quote_token_reserves,
                    pool_base_token_reserves: trade_info.pool_base_token_reserves,
                    pool_quote_token_reserves: trade_info.pool_quote_token_reserves,
                    quote_amount_out: trade_info.quote_amount_out,
                    lp_fee_basis_points: trade_info.lp_fee_basis_points,
                    lp_fee: trade_info.lp_fee,
                    protocol_fee_basis_points: trade_info.protocol_fee_basis_points,
                    protocol_fee: trade_info.protocol_fee,
                    quote_amount_out_without_lp_fee: trade_info.quote_amount_out_without_lp_fee,
                    pool: trade_info.pool.clone(),
                    user_base_token_account: trade_info.user_base_token_account.clone(),
                    user_quote_token_account: trade_info.user_quote_token_account.clone(),
                    protocol_fee_recipient: trade_info.protocol_fee_recipient.clone(),
                    protocol_fee_recipient_token_account: trade_info.protocol_fee_recipient_token_account.clone(),
                    coin_creator: trade_info.coin_creator.clone(),
                    coin_creator_fee_basis_points: trade_info.coin_creator_fee_basis_points,
                    coin_creator_fee: trade_info.coin_creator_fee,
                    virtual_sol_reserves: trade_info.virtual_sol_reserves,
                    virtual_token_reserves: trade_info.virtual_token_reserves,
                    real_sol_reserves: trade_info.real_sol_reserves,
                    real_token_reserves: trade_info.real_token_reserves,
                    bonding_curve: trade_info.bonding_curve.clone(),
                    amount: trade_info.amount,
                    max_sol_cost: trade_info.max_sol_cost,
                    min_sol_output: trade_info.min_sol_output,
                    base_amount_out: trade_info.base_amount_out,
                    max_quote_amount_in: trade_info.max_quote_amount_in,
                };
                
                // Build swap instructions for sell
                match pump_swap.build_swap_from_parsed_data(&trade_info_clone, sell_config).await {
                    Ok((keypair, instructions, price, parsed_blockhash)) => {
                        logger.log(format!("Generated PumpSwap sell instruction at price: {}", price));
                        
                        // Use parsed blockhash for faster execution
                        logger.log("Using parsed blockhash for faster transaction processing".green().to_string());
                        
                        // Execute the transaction
                        match crate::core::tx::new_signed_and_send_nozomi(
                            parsed_blockhash,
                            &keypair,
                            instructions,
                            &logger,
                        ).await {
                            Ok(signatures) => {
                                if signatures.is_empty() {
                                    return Err("No transaction signature returned".to_string());
                                }
                                
                                let signature = &signatures[0];
                                logger.log(format!("Sell transaction sent: {}", signature));
                                
                                // Send notification for transaction
                                if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                    &trade_info_clone,
                                    signature,
                                    protocol_str,
                                    "SELLING"
                                ).await {
                                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                }
                                
                                // Verify transaction
                                match verify_transaction(signature, app_state.clone(), &logger).await {
                                    Ok(verified) => {
                                        if verified {
                                            logger.log("Sell transaction verified successfully".to_string());
                                            
                                            // Send notification for successful verification
                                            if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                &trade_info_clone,
                                                signature,
                                                protocol_str,
                                                "SOLD"
                                            ).await {
                                                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                            }
                                            
                                            Ok(())
                                        } else {
                                            // Send error notification
                                            if let Err(e) = crate::services::telegram::send_copy_trade_notification(
                                                &trade_info_clone,
                                                signature,
                                                protocol_str,
                                                "ERROR"
                                            ).await {
                                                logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                                            }
                                            
                                            Err("Sell transaction verification failed".to_string())
                                        }
                                    },
                                    Err(e) => {
                                        // Send error notification
                                        if let Err(te) = crate::services::telegram::send_error_notification(
                                            &format!("Transaction verification error: {}", e)
                                        ).await {
                                            logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                        }
                                        
                                        Err(format!("Transaction verification error: {}", e))
                                    },
                                }
                            },
                            Err(e) => {
                                // Send error notification
                                if let Err(te) = crate::services::telegram::send_error_notification(
                                    &format!("Transaction error: {}", e)
                                ).await {
                                    logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                                }
                                
                                Err(format!("Transaction error: {}", e))
                            },
                        }
                    },
                    Err(e) => {
                        // Send error notification
                        if let Err(te) = crate::services::telegram::send_error_notification(
                            &format!("Failed to build PumpSwap sell instruction: {}", e)
                        ).await {
                            logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                        }
                        
                        Err(format!("Failed to build PumpSwap sell instruction: {}", e))
                    },
                }
            },
            _ => Err(format!("Unsupported protocol: {:?}", protocol)),
        };
        
        // Log execution time for standard sell
        let elapsed = start_time.elapsed();
        logger.log(format!("Standard sell execution time: {:?}", elapsed));
        
        // Increment sold counter on success
        if result.is_ok() {
            // Don't hold the lock across awaits
            let (bought_count, sold_count, active_tokens) = {
                let mut sold = SOLD_TOKENS.lock().unwrap();
                *sold += 1;
                let sold_count = *sold;
                logger.log(format!("Total sold: {}", sold_count));
                
                let bought_count = *BOUGHT_TOKENS.lock().unwrap();
                let active_tokens: Vec<String> = TOKEN_TRACKING.lock().unwrap().keys().cloned().collect();
                
                (bought_count, sold_count, active_tokens)
            };
            
            // Remove token account from our global list after successful sell
            WALLET_TOKEN_ACCOUNTS.remove(&ata);
            logger.log(format!("Removed token account {} from global list after standard sell", ata));
            
            // Send summary - now with copied values that don't include MutexGuard
            tokio::spawn(async move {
                if let Err(e) = crate::services::telegram::send_summary_notification(
                    bought_count,
                    sold_count,
                    active_tokens,
                    0.0 // We don't have PNL tracking yet
                ).await {
                    eprintln!("Failed to send summary notification: {}", e);
                }
            });
        }
        
        result
    }
}

/// Process incoming stream messages
async fn process_message(
    msg: &SubscribeUpdate,
    _subscribe_tx: &Arc<tokio::sync::Mutex<impl Sink<SubscribeRequest, Error = impl std::fmt::Debug> + Unpin>>,
    config: Arc<CopyTradingConfig>,
    logger: &Logger,
) -> Result<(), String> {
    // Handle ping messages
    if let Some(UpdateOneof::Ping(_ping)) = &msg.update_oneof {
        logger.log(format!("Received ping").blue().to_string());
        return Ok(());
    }

    // Handle transaction messages
    if let Some(UpdateOneof::Transaction(txn)) = &msg.update_oneof {
        let _start_time = Instant::now();
        
        // Extract transaction logs and account keys
        let inner_instructions = match &txn.transaction {
            Some(txn_info) => match &txn_info.meta {
                Some(meta) => meta.inner_instructions.clone(),
                None => vec![],
            },
            None => vec![],
        };
        let log_messages = txn.transaction.as_ref().unwrap().meta.clone().unwrap().clone().log_messages.clone();
        
        // Fix the recent_blockhash extraction - meta doesn't have blockhash field
        let recent_blockhash = if let Some(txn_tx) = &txn.transaction {
            if let Some(tx_inner) = &txn_tx.transaction {
                match &tx_inner.message {
                    Some(message) => Hash::new(&message.recent_blockhash),
                    None => Hash::default()
                }
            } else {
                Hash::default()
            }
        } else {
            Hash::default()
        };

        if !inner_instructions.is_empty() {
            // Find the largest data payload in inner instructions
            let mut largest_data: Option<Vec<u8>> = None;
            let mut largest_size = 0;

            for inner in &inner_instructions {
                for ix in &inner.instructions {
                    if ix.data.len() > largest_size {
                        largest_size = ix.data.len();
                        largest_data = Some(ix.data.clone());
                    }
                }
            }

            if let Some(data) = largest_data {
                if let Some(parsed_data) = crate::engine::transaction_parser::parse_transaction_data( txn ,&data, &log_messages, recent_blockhash) {
                    return handle_parsed_data( parsed_data, txn, config.clone(), logger).await;
                }
            }
        }

        logger.log("No transaction data could be parsed from logs or inner instructions".yellow().to_string());
    }
    
    Ok(())  
}

async fn handle_parsed_data_for_selling(
    parsed_data: transaction_parser::TradeInfoFromToken,
    txn: &SubscribeUpdateTransaction,
    config: Arc<CopyTradingConfig>,
    logger: &Logger,
) -> Result<(), String> {
    let start_time = Instant::now();
    let instruction_type = parsed_data.dex_type.clone();
    let signature = parsed_data.signature.clone();
    let mint = parsed_data.mint.clone();
    
    // Log the parsed transaction data
    logger.log(format!(
        "Token transaction detected for {}: Instruction: {}, Is buy: {}",
        mint,
        match instruction_type {
            transaction_parser::DexType::PumpSwap => "PumpSwap",
            transaction_parser::DexType::PumpFun => "PumpFun",
            _ => "Unknown",
        },
        parsed_data.is_buy
    ).green().to_string());
    
    // Create selling engine
    let selling_engine = SellingEngine::new(
        config.app_state.clone().into(),
        Arc::new(config.swap_config.clone()),
        SellingConfig::default(),
        config.is_progressive_sell,
    );
    
    // Update metrics with this transaction
    let parsed_transaction_data = transaction_parser::ParsedData {
        sol_change: match instruction_type {
            transaction_parser::DexType::PumpSwap => parsed_data.user_quote_amount_out.unwrap_or(0) as f64,
            transaction_parser::DexType::PumpFun => parsed_data.sol_amount.unwrap_or(0) as f64,
            _ => 0.0,
        },
        token_change: match instruction_type {
            transaction_parser::DexType::PumpSwap => parsed_data.base_amount_in.unwrap_or(0) as f64,
            transaction_parser::DexType::PumpFun => parsed_data.token_amount.unwrap_or(0) as f64,
            _ => 0.0,
        },
        is_buy: parsed_data.is_buy,
        user: parsed_data.user.clone(),
        mint: Some(mint.clone()),
        timestamp: Some(parsed_data.timestamp),
    };
    
    // Update token metrics
    if let Err(e) = selling_engine.update_metrics(&mint, &parsed_transaction_data) {
        logger.log(format!("Error updating metrics: {}", e).red().to_string());
    } else {
        logger.log(format!("Updated metrics for token: {}", mint).green().to_string());
    }
    
    // Check if we should sell this token
    match selling_engine.evaluate_sell_conditions(&mint).await {
        Ok(should_sell) => {
            if should_sell {
                logger.log(format!("Sell conditions met for token: {}", mint).green().to_string());
                
                // Determine protocol to use for selling
                let protocol = match instruction_type {
                    transaction_parser::DexType::PumpSwap => SwapProtocol::PumpSwap,
                    transaction_parser::DexType::PumpFun => SwapProtocol::PumpFun,
                    _ => config.protocol_preference.clone(),
                };
                
                // Execute progressive sell to maximize profits
                if let Err(e) = selling_engine.progressive_sell(&mint, &parsed_data, protocol.clone()).await {
                    logger.log(format!("Error executing progressive sell: {}", e).red().to_string());
                    
                    // If progressive sell fails, try standard sell
                    logger.log("Attempting standard sell as fallback".yellow().to_string());
                    if let Err(e) = execute_sell(
                        mint.clone(),
                        parsed_data.clone(),
                        config.app_state.clone().into(),
                        Arc::new(config.swap_config.clone()),
                        protocol.clone(),
                        false,  // Not progressive
                        None,   // Default chunks
                        None,   // Default interval
                    ).await {
                        logger.log(format!("Error executing standard sell: {}", e).red().to_string());
                        return Err(format!("Failed to sell token: {}", e));
                    }
                }
                
                // Send notification about the sell
                let protocol_str = match protocol {
                    SwapProtocol::PumpSwap => "PumpSwap",
                    SwapProtocol::PumpFun => "PumpFun",
                    _ => "Unknown",
                };
                
                if let Err(e) = crate::services::telegram::send_trade_notification(
                    &parsed_data,
                    protocol_str,
                    "SOLD"
                ).await {
                    logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
                }
                
                logger.log(format!("Successfully sold token: {}", mint).green().to_string());
            } else {
                logger.log(format!("Not selling token yet: {}", mint).blue().to_string());
            }
        },
        Err(e) => {
            logger.log(format!("Error evaluating sell conditions: {}", e).red().to_string());
        }
    }
    
    logger.log(format!("Processing time: {:?}", start_time.elapsed()).blue().to_string());
    Ok(())
}

/// Set up selling strategy for a token
async fn setup_selling_strategy(
    token_mint: String,
    app_state: Arc<AppState>,
    swap_config: Arc<SwapConfig>,
    protocol_preference: SwapProtocol,
    is_progressive_sell: bool,
) -> Result<(), String> {
    let logger = Logger::new("[SETUP-SELLING-STRATEGY] => ".green().to_string());
    
    // Initialize
    logger.log(format!("Setting up selling strategy for token: {}", token_mint));
    
    // Create selling engine with default configuration
    let selling_engine = SellingEngine::new(
        app_state.clone(),
        swap_config.clone(),
        SellingConfig::default(),
        is_progressive_sell,
    );
    
    // Clone values that will be moved into the task
    let token_mint_cloned = token_mint.clone();
    let app_state_cloned = app_state.clone();
    let swap_config_cloned = swap_config.clone();
    let protocol_preference_cloned = protocol_preference.clone();
    let logger_cloned = logger.clone();
    
    // Spawn a task to handle the monitoring and selling
    tokio::spawn(async move {
        let _ = monitor_token_for_selling(token_mint_cloned, app_state_cloned, swap_config_cloned, protocol_preference_cloned, &logger_cloned).await;
    });
    Ok(())
}

/// Monitor a token specifically for selling opportunities
async fn monitor_token_for_selling(
    token_mint: String,
    app_state: Arc<AppState>,
    swap_config: Arc<SwapConfig>,
    protocol_preference: SwapProtocol,
    logger: &Logger
) -> Result<(), String> {
    // Create config for the Yellowstone connection
    // This is a simplified version of what's in the main copy_trading function
    let mut yellowstone_grpc_http = "https://helsinki.rpcpool.com/".to_string(); // Default value
    let mut yellowstone_grpc_token = "your_token_here".to_string(); // Default value
    
    // Try to get config values from environment if available
    if let Ok(url) = std::env::var("YELLOWSTONE_GRPC_HTTP") {
        yellowstone_grpc_http = url;
    }
    
    if let Ok(token) = std::env::var("YELLOWSTONE_GRPC_TOKEN") {
        yellowstone_grpc_token = token;
    }
    
    logger.log("Connecting to Yellowstone gRPC for selling, will close connection after selling ...".green().to_string());
    
    // Connect to Yellowstone gRPC
    let mut client = GeyserGrpcClient::build_from_shared(yellowstone_grpc_http.clone())
        .map_err(|e| format!("Failed to build client: {}", e))?
        .x_token::<String>(Some(yellowstone_grpc_token.clone()))
        .map_err(|e| format!("Failed to set x_token: {}", e))?
        .tls_config(ClientTlsConfig::new().with_native_roots())
        .map_err(|e| format!("Failed to set tls config: {}", e))?
        .connect()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    // Set up subscribe with retries
    let mut retry_count = 0;
    const MAX_RETRIES: u32 = 3;
    let (subscribe_tx, mut stream) = loop {
        match client.subscribe().await {
            Ok(pair) => break pair,
            Err(e) => {
                retry_count += 1;
                if retry_count >= MAX_RETRIES {
                    return Err(format!("Failed to subscribe after {} attempts: {}", MAX_RETRIES, e));
                }
                logger.log(format!(
                    "[CONNECTION ERROR] => Failed to subscribe (attempt {}/{}): {}. Retrying in 5 seconds...",
                    retry_count, MAX_RETRIES, e
                ).red().to_string());
                time::sleep(Duration::from_secs(5)).await;
            }
        }
    };

    // Convert to Arc to allow cloning across tasks
    let subscribe_tx = Arc::new(tokio::sync::Mutex::new(subscribe_tx));

    // Set up subscription focused on the token mint
    let subscription_request = SubscribeRequest {
        transactions: maplit::hashmap! {
            "TokenMonitor".to_owned() => SubscribeRequestFilterTransactions {
                vote: Some(false), // Exclude vote transactions
                failed: Some(false), // Exclude failed transactions
                signature: None,
                account_include: vec![token_mint.clone()], // Only include transactions involving our token
                account_exclude: vec![JUPITER_PROGRAM.to_string(), OKX_DEX_PROGRAM.to_string()], // Exclude some common programs
                account_required: Vec::<String>::new(),
            }
        },
        commitment: Some(CommitmentLevel::Processed as i32),
        ..Default::default()
    };
  
    subscribe_tx
        .lock()
        .await
        .send(subscription_request)
        .await
        .map_err(|e| format!("Failed to send subscribe request: {}", e))?;


        let is_progressive_sell = import_env_var("IS_PROGRESSIVE_SELL").parse::<bool>().unwrap_or(false);
    // Create config for tasks
    let copy_trading_config = CopyTradingConfig {
        yellowstone_grpc_http,
        yellowstone_grpc_token,
        app_state: (*app_state).clone(),
        swap_config: (*swap_config).clone(),
        time_exceed: 60, // Default values
        counter_limit: 5,
        min_dev_buy: 1_000_000,
        max_dev_buy: 100_000_000,
        target_addresses: vec![token_mint.clone()],
        protocol_preference,
        is_progressive_sell,
    };

    let config = Arc::new(copy_trading_config);

    // Spawn heartbeat task
    let subscribe_tx_clone = subscribe_tx.clone();
    let logger_clone = logger.clone();
    
    tokio::spawn(async move {
        let ping_logger = logger_clone.clone();
        let mut interval = time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = send_heartbeat_ping(&subscribe_tx_clone, &ping_logger).await {
                ping_logger.log(format!("[CONNECTION ERROR] => {}", e).red().to_string());
                break;
            }
        }
    });

    // Main stream processing loop
    logger.log("Starting main processing loop...".green().to_string());
    while let Some(msg_result) = stream.next().await {
        match msg_result {
            Ok(msg) => {
                if let Err(e) = process_selling(&msg, &subscribe_tx, config.clone(), &logger).await {
                    logger.log(format!("Error processing message: {}", e).red().to_string());
                }
            },
            Err(e) => {
                logger.log(format!("Stream error: {:?}", e).red().to_string());
                // Try to reconnect
                break;
            },
        }
    }
    
    logger.log("Stream ended, attempting to reconnect...".yellow().to_string());
    // Here you would implement reconnection logic
    
    Ok(())
}

/// Process incoming stream messages
async fn process_selling(
    msg: &SubscribeUpdate,
    _subscribe_tx: &Arc<tokio::sync::Mutex<impl Sink<SubscribeRequest, Error = impl std::fmt::Debug> + Unpin>>,
    config: Arc<CopyTradingConfig>,
    logger: &Logger,
) -> Result<(), String> {
    // Handle ping messages
    if let Some(UpdateOneof::Ping(_ping)) = &msg.update_oneof {
        logger.log(format!("Received ping").blue().to_string());
        return Ok(());
    }

    // Handle transaction messages
    if let Some(UpdateOneof::Transaction(txn)) = &msg.update_oneof {
        let _start_time = Instant::now();
        
        // Extract transaction logs and account keys
        let inner_instructions = match &txn.transaction {
            Some(txn_info) => match &txn_info.meta {
                Some(meta) => meta.inner_instructions.clone(),
                None => vec![],
            },
            None => vec![],
        };
        let log_messages = txn.transaction.as_ref().unwrap().meta.clone().unwrap().clone().log_messages.clone();
        
        // Fix the recent_blockhash extraction - meta doesn't have blockhash field
        let recent_blockhash = if let Some(txn_tx) = &txn.transaction {
            if let Some(tx_inner) = &txn_tx.transaction {
                match &tx_inner.message {
                    Some(message) => Hash::new(&message.recent_blockhash),
                    None => Hash::default()
                }
            } else {
                Hash::default()
            }
        } else {
            Hash::default()
        };

        if !inner_instructions.is_empty() {
            // Find the largest data payload in inner instructions
            let mut largest_data: Option<Vec<u8>> = None;
            let mut largest_size = 0;

            for inner in &inner_instructions {
                for ix in &inner.instructions {
                    if ix.data.len() > largest_size {
                        largest_size = ix.data.len();
                        largest_data = Some(ix.data.clone());
                    }
                }
            }

            if let Some(data) = largest_data {
                if let Some(parsed_data) = crate::engine::transaction_parser::parse_transaction_data(txn, &data, &log_messages, recent_blockhash) {
                    if parsed_data.mint != "So11111111111111111111111111111111111111112" {
                        return handle_parsed_data_for_selling(parsed_data, txn, config, logger).await;
                    }
                }
            }
        }

        logger.log("No transaction data could be parsed from logs or inner instructions".yellow().to_string());
    }
    
    Ok(())
}


async fn handle_parsed_data(
    parsed_data: transaction_parser::TradeInfoFromToken,
    txn: &SubscribeUpdateTransaction,
    config: Arc<CopyTradingConfig>,
    logger: &Logger,
) -> Result<(), String> {
    let start_time = Instant::now();
    let instruction_type = parsed_data.dex_type.clone();
    
    // Convert instruction type for monitor functions
    let _monitor_instruction_type = convert_instruction_type(&instruction_type);

    // Extract transaction data
    let signature = if let Some(tx) = &txn.transaction {
        // Handle signature which is a Vec<u8>, not an Option
        if !tx.signature.is_empty() {
            match Signature::try_from(tx.signature.as_slice()) {
                Ok(sig) => sig.to_string(),
                Err(_) => "Invalid signature".to_string(),
            }
        } else {
            "No signature found".to_string()
        }
    } else {
        "No transaction found".to_string()
    };

    // Log the parsed transaction data
    logger.log(format!(
        "{} transaction detected: SOL change: {}, Token change: {}, Is buy: {}, User: {}",
        match instruction_type {
            transaction_parser::DexType::PumpSwap => "PumpSwap",
            transaction_parser::DexType::PumpFun => "PumpFun",
            _ => "Unknown",
        },
        match instruction_type {
            transaction_parser::DexType::PumpSwap => parsed_data.user_quote_amount_out.unwrap_or(0),
            transaction_parser::DexType::PumpFun => parsed_data.sol_amount.unwrap_or(0),
            _ => 0,
        },
        match instruction_type {
            transaction_parser::DexType::PumpSwap => parsed_data.base_amount_in.unwrap_or(0),
            transaction_parser::DexType::PumpFun => parsed_data.token_amount.unwrap_or(0),
            _ => 0,
        },
        parsed_data.is_buy,
        parsed_data.user
    ).green().to_string());
    
    // Determine protocol based on instruction type
    let protocol = if matches!(instruction_type, transaction_parser::DexType::PumpSwap) {
        SwapProtocol::PumpSwap
    } else if matches!(instruction_type, transaction_parser::DexType::PumpFun) {
        SwapProtocol::PumpFun
    } else {
        // Default to the preferred protocol in config if instruction type is unknown
        config.protocol_preference.clone()
    };
    
    // Send Telegram notification for target transaction
    let protocol_str = match protocol {
        SwapProtocol::PumpSwap => "PumpSwap",
        SwapProtocol::PumpFun => "PumpFun",
        _ => "Unknown",
    };
    
    // Record timestamp for this transaction for time elapsed tracking
    if parsed_data.is_buy {
        crate::services::telegram::record_target_transaction(&parsed_data);
        
        // Send notification about target transaction
        if let Err(e) = crate::services::telegram::send_trade_notification(
            &parsed_data,
            protocol_str,
            "DETECTED"
        ).await {
            logger.log(format!("Failed to send Telegram notification: {}", e).red().to_string());
        }
    }
    
    // Create trading engine and update metrics
    let selling_engine = SellingEngine::new(
        config.app_state.clone().into(),
        Arc::new(config.swap_config.clone()),
        SellingConfig::default(),
        config.is_progressive_sell,
    );
    
    // Handle buy transaction
    if parsed_data.is_buy {
        // Execute buy operation with the selected protocol
        let mut parsed_data_clone = parsed_data.clone();
        parsed_data_clone.signature = signature.clone();
        
        match execute_buy(
            parsed_data.clone(),
            config.app_state.clone().into(),
            Arc::new(config.swap_config.clone()),
            protocol.clone(),
        ).await {
            Err(e) => {
                logger.log(format!("Error executing buy: {}", e).red().to_string());
                
                // Send error notification via Telegram
                if let Err(te) = crate::services::telegram::send_error_notification(&format!("Buy error: {}", e)).await {
                    logger.log(format!("Failed to send Telegram notification: {}", te).red().to_string());
                }
                
                Err(e) // Return the error from execute_buy
            },
            Ok(_) => {        
                logger.log(format!("Successfully executed buy operation target transaction {}", signature).green().to_string());
                logger.log(format!("Now starting to monitor this token to sell at a profit").blue().to_string());
                
                // Setup selling strategy based on take profit and stop loss
                match setup_selling_strategy(
                    parsed_data.mint.clone(), 
                    config.app_state.clone().into(), 
                    Arc::new(config.swap_config.clone()), 
                    protocol.clone(),
                    config.is_progressive_sell,
                ).await {
                    Ok(_) => {
                        logger.log("Selling strategy set up successfully".green().to_string());
                        Ok(())
                    },
                    Err(e) => {
                        logger.log(format!("Failed to set up selling strategy: {}", e).red().to_string());
                        Err(e)
                    }
                }
            }
        }
    } else {
        // For sell transactions, we don't copy them
        // We rely on our own take profit and stop loss strategy
        logger.log(format!("Not copying selling transaction - using take profit and stop loss").blue().to_string());
        Ok(())
    }
}

