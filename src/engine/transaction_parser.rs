use bs58;
use std::str::FromStr;
use std::collections::HashMap;
use anchor_client::solana_sdk::hash::Hash;
use solana_sdk::pubkey::Pubkey;
use serde::{Deserialize, Serialize};
use colored::Colorize;
use crate::common::logger::Logger;
use base64;
use lazy_static;
use yellowstone_grpc_proto::geyser::SubscribeUpdateTransaction;
use anyhow::{Result, anyhow};
// Create a static logger for this module
lazy_static::lazy_static! {
    static ref LOGGER: Logger = Logger::new("[TRANSACTION-PARSER] => ".blue().to_string());
}

#[derive(Clone, Debug, PartialEq)]
pub enum DexType {
    PumpSwap,
    PumpFun,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct ParsedData {
    pub sol_change: f64,
    pub token_change: f64,
    pub is_buy: bool,
    pub user: String,
    pub mint: Option<String>,
    pub timestamp: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PumpSwapData {
    pub timestamp: u64,
    pub base_amount_in: u64,
    pub min_quote_amount_out: u64,
    pub user_base_token_reserves: u64,
    pub user_quote_token_reserves: u64,
    pub pool_base_token_reserves: u64,
    pub pool_quote_token_reserves: u64,
    pub quote_amount_out: u64,
    pub lp_fee_basis_points: u64,
    pub lp_fee: u64,
    pub protocol_fee_basis_points: u64,
    pub protocol_fee: u64,
    pub quote_amount_out_without_lp_fee: u64,
    pub user_quote_amount_out: u64,
    pub pool: String,
    pub user: String,
    pub target_user_base_token_account: String,
    pub target_user_quote_token_account: String,
    pub protocol_fee_recipient: String,
    pub protocol_fee_recipient_token_account: String,
}

#[derive(Clone, Debug)]
pub struct PumpFunData {
    pub mint: String,
    pub sol_amount: u64,
    pub token_amount: u64,
    pub is_buy: bool,
    pub user: String,
    pub timestamp: u64,
    pub virtual_sol_reserves: u64,
    pub virtual_token_reserves: u64,
    pub real_sol_reserves: u64,
    pub real_token_reserves: u64,
}

#[derive(Clone, Debug)]
pub struct TransactionOutput {
    pub sol_changes: f64,
    pub token_changes: f64,
    pub is_buy: bool,
    pub user: String,
    pub instruction_type: DexType,
    pub timestamp: u64,
    pub mint: String,
    pub signature: String,
}

#[derive(Clone, Debug)]
pub struct TradeInfoFromToken {
    // Common fields
    pub dex_type: DexType,
    pub slot: u64,
    pub recent_blockhash: Hash,
    pub signature: String,
    pub target: String,
    // Fields from both PumpSwapData and PumpFunData
    pub mint: String,
    pub user: String,
    pub timestamp: u64,
    pub is_buy: bool,
    // PumpSwapData fields
    pub base_amount_in: Option<u64>,
    pub min_quote_amount_out: Option<u64>,
    pub user_base_token_reserves: Option<u64>,
    pub user_quote_token_reserves: Option<u64>,
    pub pool_base_token_reserves: Option<u64>,
    pub pool_quote_token_reserves: Option<u64>,
    pub quote_amount_out: Option<u64>, //  quote_amount_out in case of sell , quote_amount_in in case of buy
    pub lp_fee_basis_points: Option<u64>,
    pub lp_fee: Option<u64>,
    pub protocol_fee_basis_points: Option<u64>,
    pub protocol_fee: Option<u64>,
    pub quote_amount_out_without_lp_fee: Option<u64>,
    pub user_quote_amount_out: Option<u64>, // user_quote_amount_out in case of sell , user_base_amount_in in case of buy
    pub pool: Option<String>,
    pub user_base_token_account: Option<String>,
    pub user_quote_token_account: Option<String>,
    pub protocol_fee_recipient: Option<String>,
    pub protocol_fee_recipient_token_account: Option<String>,
    pub coin_creator: Option<String>,
    pub coin_creator_fee_basis_points: Option<u64>,
    pub coin_creator_fee: Option<u64>,
    
    // PumpFunData fields
    pub sol_amount: Option<u64>,
    pub token_amount: Option<u64>,
    pub virtual_sol_reserves: Option<u64>,
    pub virtual_token_reserves: Option<u64>,
    pub real_sol_reserves: Option<u64>,
    pub real_token_reserves: Option<u64>,
    
    // Additional fields from original TradeInfoFromToken
    pub bonding_curve: String,
    pub volume_change: i64,
    pub bonding_curve_info: Option<crate::engine::monitor::BondingCurveInfo>,
    pub pool_info: Option<crate::engine::monitor::PoolInfo>,
    pub token_amount_f64: f64,
    pub amount: Option<u64>,
    pub max_sol_cost: Option<u64>,
    pub min_sol_output: Option<u64>,
    pub base_amount_out: Option<u64>,
    pub max_quote_amount_in: Option<u64>,
}


/// Parses the transaction data buffer into a TradeInfoFromToken struct
pub fn parse_transaction_data(txn: &SubscribeUpdateTransaction, buffer: &[u8], log_messages: &[String], recent_blockhash: Hash) -> Option<TradeInfoFromToken> {
    fn parse_public_key(buffer: &[u8], offset: usize) -> Option<String> {
        if offset + 32 > buffer.len() {
            return None;
        }
        Some(bs58::encode(&buffer[offset..offset+32]).into_string())
    }

    fn parse_u64(buffer: &[u8], offset: usize) -> Option<u64> {
        if offset + 8 > buffer.len() {
            return None;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buffer[offset..offset+8]);
        Some(u64::from_le_bytes(bytes))
    }
    
    // Helper function to extract token mint from token balances
    fn extract_token_info(
        txn: &SubscribeUpdateTransaction,
        log_messages: &[String],
    ) -> Option<String> {
        // Default value
        let mut mint = String::new();
        
        // Try to extract from token balances if txn is available
        if let Some(tx_inner) = &txn.transaction {
            if let Some(meta) = &tx_inner.meta {
                // Check post token balances
                if !meta.post_token_balances.is_empty() {
                    mint = meta.post_token_balances[0].mint.clone();
                }
            }
        }
        // If we couldn't extract from token balances, use default
        if mint.is_empty() {
            mint = "2ivzYvjnKqA4X3dVvPKr7bctGpbxwrXbbxm44TJCpump".to_string();
        }
        
        Some(mint)
    }

    let mut mint = String::new();
    println!("buffer: {:?}", buffer.len());
    
    // No transaction available in this context, so we'll pass None
    match buffer.len() {
        368 => {
            // Extract token mint - pass None for the txn since we don't have it here
            let mint = extract_token_info(&txn, log_messages).unwrap_or_else(|| 
                "2ivzYvjnKqA4X3dVvPKr7bctGpbxwrXbbxm44TJCpump".to_string()
            );
            
            let timestamp = parse_u64(buffer, 16)?;
            let base_amount_in = parse_u64(buffer, 24)?;
            let min_quote_amount_out = parse_u64(buffer, 32)?;
            let user_base_token_reserves = parse_u64(buffer, 40)?;
            let user_quote_token_reserves = parse_u64(buffer, 48)?;
            let pool_base_token_reserves = parse_u64(buffer, 56)?;
            let pool_quote_token_reserves = parse_u64(buffer, 64)?;
            let quote_amount_out = parse_u64(buffer, 72)?;
            let lp_fee_basis_points = parse_u64(buffer, 80)?;
            let lp_fee = parse_u64(buffer, 88)?;
            let protocol_fee_basis_points = parse_u64(buffer, 96)?;
            let protocol_fee = parse_u64(buffer, 104)?;
            let quote_amount_out_without_lp_fee = parse_u64(buffer, 112)?;
            let user_quote_amount_out = parse_u64(buffer, 120)?;
            let pool = parse_public_key(buffer, 128)?;
            let user = parse_public_key(buffer, 160)?;
            let user_base_token_account = parse_public_key(buffer, 192)?;
            let user_quote_token_account = parse_public_key(buffer, 224)?;
            let protocol_fee_recipient = parse_public_key(buffer, 256)?;
            let protocol_fee_recipient_token_account = parse_public_key(buffer, 288)?;
            let coin_creator = parse_public_key(buffer, 320)?;
            let coin_creator_fee_basis_points = parse_u64(buffer, 328)?;
            let coin_creator_fee = parse_u64(buffer, 336)?;
            let is_buy = quote_amount_out_without_lp_fee > quote_amount_out;

            let pool_info = crate::engine::monitor::PoolInfo {
                pool_id: Pubkey::from_str(&pool).unwrap_or_else(|_| panic!("Invalid pool pubkey: {}", pool)),
                base_mint: Pubkey::from_str(&mint).unwrap_or_else(|_| panic!("Invalid mint pubkey: {}", mint)),
                quote_mint: Pubkey::from_str("So11111111111111111111111111111111111111112").unwrap_or_else(|_| panic!("Invalid WSOL mint pubkey")),
                base_reserve: pool_base_token_reserves,
                quote_reserve: pool_quote_token_reserves,
                coin_creator: Pubkey::from_str(&coin_creator).unwrap_or_else(|_| panic!("Invalid creator pubkey: {}", coin_creator)),
            };

            Some(TradeInfoFromToken {
                dex_type: DexType::PumpSwap,
                slot: 0, // Will be set from transaction data
                recent_blockhash, // Will be set from transaction data
                signature: String::new(), // Will be set from transaction data
                target: String::new(), // Will be set from transaction data
                mint, // Will be set from pool data
                user,
                timestamp,
                is_buy,
                // PumpSwapData fields
                base_amount_in: Some(base_amount_in),
                min_quote_amount_out: Some(min_quote_amount_out),
                user_base_token_reserves: Some(user_base_token_reserves),
                user_quote_token_reserves: Some(user_quote_token_reserves),
                pool_base_token_reserves: Some(pool_base_token_reserves),
                pool_quote_token_reserves: Some(pool_quote_token_reserves),
                quote_amount_out: Some(quote_amount_out),
                lp_fee_basis_points: Some(lp_fee_basis_points),
                lp_fee: Some(lp_fee),
                protocol_fee_basis_points: Some(protocol_fee_basis_points),
                protocol_fee: Some(protocol_fee),
                quote_amount_out_without_lp_fee: Some(quote_amount_out_without_lp_fee),
                user_quote_amount_out: Some(user_quote_amount_out),
                pool: Some(pool),
                user_base_token_account: Some(user_base_token_account),
                user_quote_token_account: Some(user_quote_token_account),
                protocol_fee_recipient: Some(protocol_fee_recipient),
                protocol_fee_recipient_token_account: Some(protocol_fee_recipient_token_account),
                coin_creator: Some(coin_creator),
                coin_creator_fee_basis_points: Some(coin_creator_fee_basis_points),
                coin_creator_fee: Some(coin_creator_fee),

                // PumpFunData fields - None
                sol_amount: None,
                token_amount: None,
                virtual_sol_reserves: None,
                virtual_token_reserves: None,
                real_sol_reserves: None,
                real_token_reserves: None,
                
                // Additional fields
                bonding_curve: String::new(),
                volume_change: 0,
                bonding_curve_info: None,
                pool_info: Some(pool_info),
                token_amount_f64: base_amount_in as f64,
                amount: None,
                max_sol_cost: None,
                min_sol_output: None,
                base_amount_out: None,
                max_quote_amount_in: None,
            })
        },
        185 => {
            // Parse PumpFunData fields
            let mint = parse_public_key(buffer, 16)?;
            let sol_amount = parse_u64(buffer, 48)?;
            let token_amount = parse_u64(buffer, 56)?;
            let is_buy = buffer.get(64)? == &1;
            let user = parse_public_key(buffer, 65)?;
            let timestamp = parse_u64(buffer, 97)?;
            let virtual_sol_reserves = parse_u64(buffer, 105)?;
            let virtual_token_reserves = parse_u64(buffer, 113)?;
            let real_sol_reserves = parse_u64(buffer, 121)?;
            let real_token_reserves = parse_u64(buffer, 129)?;
            let creator = parse_public_key(buffer, 137)?;
            let creator_fee_basis_points = parse_u64(buffer, 169)?;
            let creator_fee = parse_u64(buffer, 177)?;


            Some(TradeInfoFromToken {
                dex_type: DexType::PumpFun,
                slot: 0, // Will be set from transaction data
                recent_blockhash, // Will be set from transaction data
                signature: String::new(), // Will be set from transaction data
                target: String::new(), // Will be set from transaction data
                mint,
                user,
                timestamp,
                is_buy,
                // PumpSwapData fields - None
                base_amount_in: None,
                min_quote_amount_out: None,
                user_base_token_reserves: None,
                user_quote_token_reserves: None,
                pool_base_token_reserves: None,
                pool_quote_token_reserves: None,
                quote_amount_out: None,
                lp_fee_basis_points: None,
                lp_fee: None,
                protocol_fee_basis_points: None,
                protocol_fee: None,
                quote_amount_out_without_lp_fee: None,
                user_quote_amount_out: None,
                pool: None,
                user_base_token_account: None,
                user_quote_token_account: None,
                protocol_fee_recipient: None,
                protocol_fee_recipient_token_account: None,
                coin_creator: Some(creator),
                coin_creator_fee_basis_points: Some(creator_fee_basis_points),
                coin_creator_fee: Some(creator_fee),
                
                // PumpFunData fields
                sol_amount: Some(sol_amount),
                token_amount: Some(token_amount),
                virtual_sol_reserves: Some(virtual_sol_reserves),
                virtual_token_reserves: Some(virtual_token_reserves),
                real_sol_reserves: Some(real_sol_reserves),
                real_token_reserves: Some(real_token_reserves),
                
                // Additional fields
                bonding_curve: String::new(),
                volume_change: 0,
                bonding_curve_info: None,
                pool_info: None,
                token_amount_f64: token_amount as f64,
                amount: None,
                max_sol_cost: None,
                min_sol_output: None,
                base_amount_out: None,
                max_quote_amount_in: None,
            })
        },
        _ => None,
    }
}

pub fn extract_pump_swap_params(trade_info: &TradeInfoFromToken) -> Result<HashMap<String, f64>, anyhow::Error> {
    let mut params = HashMap::new();
    
    if trade_info.dex_type != DexType::PumpSwap {
        return Err(anyhow!("Not a PumpSwap transaction"));
    }
    
    // Extract basic parameters
    params.insert("base_amount_in".to_string(), trade_info.base_amount_in.unwrap_or(0) as f64);
    params.insert("min_quote_amount_out".to_string(), trade_info.min_quote_amount_out.unwrap_or(0) as f64);
    params.insert("quote_amount_out".to_string(), trade_info.quote_amount_out.unwrap_or(0) as f64);
    params.insert("user_quote_amount_out".to_string(), trade_info.user_quote_amount_out.unwrap_or(0) as f64);
    
    // Calculate derived parameters
    if let (Some(base), Some(quote)) = (trade_info.pool_base_token_reserves, trade_info.pool_quote_token_reserves) {
        if base > 0 {
            let price = quote as f64 / base as f64;
            params.insert("price".to_string(), price);
            
            if let Some(amount) = trade_info.base_amount_in {
                let value = price * amount as f64;
                params.insert("value".to_string(), value);
            }
        }
    }
    
    Ok(params)
}

pub fn extract_pump_fun_params(trade_info: &TradeInfoFromToken) -> Result<HashMap<String, f64>, anyhow::Error> {
    let mut params = HashMap::new();
    
    if trade_info.dex_type != DexType::PumpFun {
        return Err(anyhow!("Not a PumpFun transaction"));
    }
    
    // Extract basic parameters
    params.insert("sol_amount".to_string(), trade_info.sol_amount.unwrap_or(0) as f64);
    params.insert("token_amount".to_string(), trade_info.token_amount.unwrap_or(0) as f64);
    
    // Add virtual reserves if available
    if let Some(virtual_sol) = trade_info.virtual_sol_reserves {
        params.insert("virtual_sol_reserves".to_string(), virtual_sol as f64);
    }
    
    if let Some(virtual_token) = trade_info.virtual_token_reserves {
        params.insert("virtual_token_reserves".to_string(), virtual_token as f64);
    }
    
    // Calculate price if possible
    if let (Some(vsol), Some(vtoken)) = (trade_info.virtual_sol_reserves, trade_info.virtual_token_reserves) {
        if vtoken > 0 {
            let price = vsol as f64 / vtoken as f64;
            params.insert("price".to_string(), price);
            
            if let Some(amount) = trade_info.token_amount {
                let value = price * amount as f64;
                params.insert("value".to_string(), value);
            }
        }
    }
    
    Ok(params)
}

/// Extract transaction data from logs by searching for encoded data in logs
/// Returns the largest byte array found in any "Program data:" line
pub fn extract_transaction_data_from_inner_instructions(log_messages: &[String]) -> Option<Vec<u8>> {
    // Look for program data lines
    let mut largest_data: Option<Vec<u8>> = None;
    let mut largest_size = 0;
    
    for log in log_messages {
        // Try to find program data lines that contain encoded transaction data
        if log.contains("Program data:") {
            // Extract the encoded data part after "Program data:"
            let parts: Vec<&str> = log.split("Program data:").collect();
            if parts.len() > 1 {
                let encoded_data = parts[1].trim();
                // Decode the base58 or base64 data
                if let Ok(decoded) = bs58::decode(encoded_data).into_vec() {
                    if decoded.len() > largest_size {
                        largest_size = decoded.len();
                        largest_data = Some(decoded);
                    }
                }
                // Try to handle other encodings if bs58 failed
                else if let Ok(decoded) = base64::decode(encoded_data) {
                    if decoded.len() > largest_size {
                        largest_size = decoded.len();
                        largest_data = Some(decoded);
                    }
                }
            }
        }
        // Also check for lines that might contain raw hex data
        else if log.contains("Pi83CqUD") || log.contains("3Upy1xAC") {
            // These prefixes might indicate PumpSwap or PumpFun encoded data
            let encoded_part = log.trim();
            if let Ok(decoded) = bs58::decode(encoded_part).into_vec() {
                if decoded.len() > largest_size {
                    largest_size = decoded.len();
                    largest_data = Some(decoded);
                }
            }
        }
    }
    
    largest_data
}

/// Process a decoded instruction to extract transaction data
pub fn process_decoded_instruction(
    _decoded_ix: HashMap<String, serde_json::Value>,
    log_messages: &[String],
    recent_blockhash: Hash,
) -> Option<TradeInfoFromToken> {
    let _mint = String::new(); // Rename to _mint since we don't use it yet
    
    // Extract any transaction data from logs
    if let Some(data) = extract_transaction_data_from_inner_instructions(log_messages) {
        // Try to parse transaction data
        return parse_transaction_data(&SubscribeUpdateTransaction::default(), &data, log_messages, recent_blockhash);
    }
    
    None // Return None if we couldn't extract or parse anything
} 