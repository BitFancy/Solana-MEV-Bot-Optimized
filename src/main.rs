/*
 * Copy Trading Bot with PumpSwap Notification Mode
 * 
 * Changes made:
 * - Modified PumpSwap buy/sell logic to only send notifications without executing transactions
 * - Transaction processing now runs in separate tokio tasks to ensure main monitoring continues
 * - Added placeholder for future selling strategy implementation
 * - PumpFun protocol functionality remains unchanged
 * - Added caching and batch RPC calls for improved performance
 */
use anchor_client::solana_sdk::signature::Signer;
use solana_vntr_sniper::{
    common::{config::Config, constants::RUN_MSG, cache::WALLET_TOKEN_ACCOUNTS},
    engine::{
        copy_trading::{start_copy_trading, CopyTradingConfig},
        swap::SwapProtocol,
    },
    services::{telegram, cache_maintenance},
};
use anchor_client::solana_client::rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig};
use anchor_client::solana_client::rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType};
use anchor_client::solana_sdk::pubkey::Pubkey;
use solana_account_decoder::UiAccountEncoding;
use std::str::FromStr;
use colored::Colorize;

/// Initialize the wallet token account list by fetching all token accounts owned by the wallet
async fn initialize_token_account_list(config: &Config) {
    let logger = solana_vntr_sniper::common::logger::Logger::new("[INIT-TOKEN-ACCOUNTS] => ".green().to_string());
    
    if let Ok(wallet_pubkey) = config.app_state.wallet.try_pubkey() {
        logger.log(format!("Initializing token account list for wallet: {}", wallet_pubkey));
        
        // Get the token program pubkey
        let token_program = Pubkey::from_str("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA").unwrap();
        
        // Query all token accounts owned by the wallet
        let accounts = config.app_state.rpc_client.get_token_accounts_by_owner(
            &wallet_pubkey,
            anchor_client::solana_client::rpc_request::TokenAccountsFilter::ProgramId(token_program)
        );
        match accounts {
            Ok(accounts) => {
                logger.log(format!("Found {} token accounts", accounts.len()));
                
                // Add each token account to our global cache
                for account in accounts {
                    WALLET_TOKEN_ACCOUNTS.insert(Pubkey::from_str(&account.pubkey).unwrap());
                    logger.log(format!("Added token account: {}", account.pubkey ));
                }
                
                logger.log(format!("Token account list initialized with {} accounts", WALLET_TOKEN_ACCOUNTS.size()));
            },
            Err(e) => {
                logger.log(format!("Error fetching token accounts: {}", e));
            }
        }
    } else {
        logger.log("Failed to get wallet pubkey, can't initialize token account list".to_string());
    }
}

#[tokio::main]
async fn main() {
    /* Initial Settings */
    let config = Config::new().await;
    let config = config.lock().await;

    /* Running Bot */
    let run_msg = RUN_MSG;
    println!("{}", run_msg);

    // Initialize Telegram bot
    match telegram::init().await {
        Ok(_) => println!("Telegram bot initialized successfully"),
        Err(e) => println!("Failed to initialize Telegram bot: {}. Continuing without notifications.", e),
    }
    
    // Initialize token account list
    initialize_token_account_list(&config).await;
    
    // Start cache maintenance service (clean up expired cache entries every 60 seconds)
    cache_maintenance::start_cache_maintenance(60).await;
    println!("Cache maintenance service started");

    // Get copy trading target addresses from environment
    let copy_trading_target_address = std::env::var("COPY_TRADING_TARGET_ADDRESS").ok();
    let is_multi_copy_trading = std::env::var("IS_MULTI_COPY_TRADING")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(false);
    
    // Prepare target addresses for monitoring
    let mut target_addresses = Vec::new();
    
    // Handle multiple copy trading targets if enabled
    if is_multi_copy_trading {
        if let Some(address_str) = copy_trading_target_address {
            // Parse comma-separated addresses
            for addr in address_str.split(',') {
                let trimmed_addr = addr.trim();
                if !trimmed_addr.is_empty() {
                    target_addresses.push(trimmed_addr.to_string());
                }
            }
        }
    } else if let Some(address) = copy_trading_target_address {
        // Single address mode
        if !address.is_empty() {
            target_addresses.push(address);
        }
    }
    
    if target_addresses.is_empty() {
        eprintln!("No COPY_TRADING_TARGET_ADDRESS specified. Please set this environment variable.");
        return;
    }
    
    // Get protocol preference from environment
    let protocol_preference = std::env::var("PROTOCOL_PREFERENCE")
        .ok()
        .map(|p| match p.to_lowercase().as_str() {
            "pumpfun" => SwapProtocol::PumpFun,
            "pumpswap" => SwapProtocol::PumpSwap,
            _ => SwapProtocol::Auto,
        })
        .unwrap_or(SwapProtocol::Auto);
    
    // Create copy trading config
    let copy_trading_config = CopyTradingConfig {
        yellowstone_grpc_http: config.yellowstone_grpc_http.clone(),
        yellowstone_grpc_token: config.yellowstone_grpc_token.clone(),
        app_state: config.app_state.clone(),
        swap_config: config.swap_config.clone(),
        time_exceed: config.time_exceed,
        counter_limit: config.counter_limit as u64,
        min_dev_buy: config.min_dev_buy as u64,
        max_dev_buy: config.max_dev_buy as u64,
        target_addresses,
        protocol_preference,
        is_progressive_sell: config.is_progressive_sell,
    };
    
    // Start the copy trading bot
    if let Err(e) = start_copy_trading(copy_trading_config).await {
        eprintln!("Copy trading error: {}", e);
        
        // Send error notification via Telegram
        if let Err(te) = telegram::send_error_notification(&format!("Copy trading bot crashed: {}", e)).await {
            eprintln!("Failed to send Telegram notification: {}", te);
        }
    }
}
