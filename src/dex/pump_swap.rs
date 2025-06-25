use std::{str::FromStr, sync::Arc};
use solana_program_pack::Pack;
use anchor_client::solana_client::rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig};
use anchor_client::solana_client::rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType};
use anchor_client::solana_client::rpc_response::RpcKeyedAccount;
use solana_account_decoder::UiAccountEncoding;
use anyhow::{anyhow, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use anchor_client::solana_sdk::{
    hash::Hash,
    instruction::{AccountMeta, Instruction},
    system_instruction,
    pubkey::Pubkey,
    signature::Keypair,
    system_program,
    signer::Signer,
};
use crate::engine::transaction_parser::DexType;
use spl_associated_token_account::{
    get_associated_token_address,
    instruction::{create_associated_token_account, create_associated_token_account_idempotent}
};
use spl_token::{amount_to_ui_amount, ui_amount_to_amount};
use tokio::time::Instant;


use crate::{
    common::{config::SwapConfig, logger::Logger, cache::{POOL_CACHE, WALLET_TOKEN_ACCOUNTS}},
    core::token,
    engine::swap::{SwapDirection, SwapInType},
};

use crate::services::rpc_client::BatchRpcClient;

// Constants
pub const TEN_THOUSAND: u64 = 10000;
pub const TOKEN_PROGRAM: &str = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA";
pub const TOKEN_2022_PROGRAM: &str = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb";
pub const ASSOCIATED_TOKEN_PROGRAM: &str = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL";
pub const PUMP_SWAP_PROGRAM: &str = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA";
pub const PUMP_GLOBAL_CONFIG: &str = "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw";
pub const PUMP_FEE_RECIPIENT: &str = "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV";
pub const PUMP_EVENT_AUTHORITY: &str = "GS4CU59F31iL7aR2Q8zVS8DRrcRnXX1yjQ66TqNVQnaR";
pub const BUY_DISCRIMINATOR: [u8; 8] = [102, 6, 61, 18, 1, 218, 235, 234];
pub const SELL_DISCRIMINATOR: [u8; 8] = [51, 230, 133, 164, 1, 127, 131, 173];
pub const SOL_MINT: &str = "So11111111111111111111111111111111111111112";

/// A struct to represent the PumpSwap pool which uses constant product AMM
#[derive(Debug, Clone)]
pub struct PumpSwapPool {
    pub pool_id: Pubkey,
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub lp_mint: Pubkey,
    pub pool_base_account: Pubkey,
    pub pool_quote_account: Pubkey,
    pub base_reserve: u64,
    pub quote_reserve: u64,
    pub coin_creator: Pubkey,
}

pub struct PumpSwap {
    pub keypair: Arc<Keypair>,
    pub rpc_client: Option<Arc<anchor_client::solana_client::rpc_client::RpcClient>>,
    pub rpc_nonblocking_client: Option<Arc<anchor_client::solana_client::nonblocking::rpc_client::RpcClient>>,
}

impl PumpSwap {
    pub fn new(
        keypair: Arc<Keypair>,
        rpc_client: Option<Arc<anchor_client::solana_client::rpc_client::RpcClient>>,
        rpc_nonblocking_client: Option<Arc<anchor_client::solana_client::nonblocking::rpc_client::RpcClient>>,
    ) -> Self {
        Self {
            keypair,
            rpc_client,
            rpc_nonblocking_client,
        }
    }

    pub async fn get_pump_swap_pool(
        &self,
        mint_str: &str,
    ) -> Result<PumpSwapPool> {
        let mint = Pubkey::from_str(mint_str).map_err(|_| anyhow!("Invalid mint address"))?;
        let rpc_client = self.rpc_client.clone()
            .ok_or_else(|| anyhow!("RPC client not initialized"))?;
        get_pool_info(rpc_client, mint).await
    }

    // Add new method to build swap instructions from parsed transaction data
    pub async fn build_swap_from_parsed_data(
        &self,
        trade_info: &crate::engine::transaction_parser::TradeInfoFromToken,
        swap_config: SwapConfig,
    ) -> Result<(Arc<Keypair>, Vec<Instruction>, f64, Hash)> {
        let logger = Logger::new("[PUMPSWAP-FROM-PARSED] => ".blue().to_string());
        logger.log(format!("Building PumpSwap transaction from parsed data"));
        
        // Basic validation - ensure we have a PumpSwap transaction
        if trade_info.dex_type != DexType::PumpSwap {
            return Err(anyhow!("Invalid transaction type, expected PumpSwap"));
        }
        
        // Extract essential data
        let mint_str = &trade_info.mint;
        let owner = self.keypair.pubkey();
        let sol_mint = Pubkey::from_str(SOL_MINT)?;
        
        // Get a fresh blockhash
        let blockhash = if trade_info.recent_blockhash != Hash::default() {
            // Use blockhash from trade_info if available
            trade_info.recent_blockhash
        } else {
            // Get fresh blockhash if not available
            match &self.rpc_client {
                Some(client) => {
                    client.get_latest_blockhash()
                        .map_err(|e| anyhow!("Failed to get blockhash: {}", e))?
                },
                None => return Err(anyhow!("RPC client not initialized")),
            }
        };
        
        // Get slippage from swap config
        let slippage_bps = swap_config.slippage;
        
        // Determine if this is a buy or sell operation
        let (token_in, token_out, discriminator) = match swap_config.swap_direction {
            SwapDirection::Buy => (sol_mint, Pubkey::from_str(mint_str)?, BUY_DISCRIMINATOR),
            SwapDirection::Sell => (Pubkey::from_str(mint_str)?, sol_mint, SELL_DISCRIMINATOR),
        };
        
        // Get PumpSwap program ID
        let pump_program = Pubkey::from_str(PUMP_SWAP_PROGRAM)?;
        
        // Create instructions as needed
        let mut create_instruction = None;
        //  In case of SwapDirection::Sell, close instruction includes wsol account close instruction 
        // Or if wallet have wsol account don't need to create wsol account or close wsol account
        // So This is just for Dev test case
        let mut close_instruction = None;
        
        // Extract or fetch pool info
        let mint = Pubkey::from_str(mint_str)?;
        let pool_info = if let Some(info) = &trade_info.pool_info {
            // Use pool info directly from trade_info
            PumpSwapPool {
                pool_id: info.pool_id,
                base_mint: token_in,
                quote_mint: token_out,
                lp_mint: Pubkey::default(),
                pool_base_account: get_associated_token_address(&info.pool_id, &mint),
                pool_quote_account: get_associated_token_address(&info.pool_id, &sol_mint),
                base_reserve: info.base_reserve,
                quote_reserve: info.quote_reserve,
                coin_creator: info.coin_creator,
            }
        } else {
            // Need to build pool info from other sources
            
            // First get pool_id
            let pool_id = if let Some(pool) = &trade_info.pool {
                Pubkey::from_str(pool)?
            } else if let Some(cached_pool) = POOL_CACHE.get(&mint) {
                cached_pool.pool_id
            } else {
                logger.log(format!("Pool info not found in parsed data or cache, querying on-chain"));
                let pool_info = get_pool_info(self.rpc_client.clone().unwrap(), mint).await?;
                pool_info.pool_id
            };
            
            // Then get pool reserves
            let (base_reserve, quote_reserve) = if let Some(base) = trade_info.pool_base_token_reserves {
                let quote = trade_info.pool_quote_token_reserves.unwrap_or(0);
                (base, quote)
            } else if let Some(cached_pool) = POOL_CACHE.get(&mint) {
                logger.log(format!("Using cached pool reserves"));
                (cached_pool.base_reserve, cached_pool.quote_reserve)
            } else {
                logger.log(format!("Pool reserves not found in parsed data or cache, querying on-chain"));
                let pool_info = get_pool_info(self.rpc_client.clone().unwrap(), mint).await?;
                (pool_info.base_reserve, pool_info.quote_reserve)
            };
            
            // Finally, either use cached pool or get complete pool info
            if let Some(cached_pool) = POOL_CACHE.get(&mint) {
                logger.log(format!("Using cached pool details"));
                cached_pool
            } else {
                logger.log(format!("Pool details not found in cache, querying on-chain"));
                get_pool_info(self.rpc_client.clone().unwrap(), mint).await?
            }
        };
        
        // Log pool reserves info
        logger.log(format!("Pool reserves - Base: {}, Quote: {}", 
                           pool_info.base_reserve, pool_info.quote_reserve));
        
        // Calculate token price
        let token_price: f64 = if pool_info.base_reserve > 0 {
            (pool_info.quote_reserve as f64) / (pool_info.base_reserve as f64)
        } else {
            0.0
        };
        logger.log(format!("Token price: {} SOL", token_price));
        
        // Get ATAs for input and output tokens
        let mut in_ata = get_associated_token_address(&owner, &token_in);
        let mut out_ata = get_associated_token_address(&owner, &token_out);
        
        // Check token accounts using our global tracker instead of RPC calls
        let in_ata_exists = WALLET_TOKEN_ACCOUNTS.contains(&in_ata);
        let out_ata_exists = WALLET_TOKEN_ACCOUNTS.contains(&out_ata);
        logger.log(format!("Token account status from global tracker - In: {}, Out: {}", 
                         in_ata_exists, out_ata_exists));

        let wsol_token_account = get_associated_token_address(&owner, &sol_mint);
        
        // Process based on swap direction
        let (base_amount, quote_amount, accounts) = match swap_config.swap_direction {
            SwapDirection::Buy => {
                // Create token account if needed
                if !out_ata_exists {
                    logger.log(format!("Output token account {} doesn't exist, adding creation instruction", out_ata));
                    create_instruction = Some(create_associated_token_account_idempotent(
                        &owner,
                        &owner,
                        &token_out,
                        &Pubkey::from_str(TOKEN_PROGRAM)?,
                    ));
                    
                    // Add to our tracking list - it will exist after transaction
                    WALLET_TOKEN_ACCOUNTS.insert(out_ata);
                    logger.log(format!("Added {} to token account tracker (will exist after transaction)", out_ata));
                } else {
                    logger.log(format!("Output token account {} already exists", out_ata));
                }
                
                // Always initialize WSOL account for buys when using SOL as input
                if token_in == sol_mint {
                    // Get the WSOL token account address (ATA)
                    logger.log(format!("WSOL token account: {}", wsol_token_account));
                    
                    // Calculate amount with slippage for the transfer
                    in_ata = wsol_token_account;
                }
                // Calculate expected token amount
                let amount_specified =ui_amount_to_amount(swap_config.amount_in, 9); 
                let base_amount_out = calculate_buy_base_amount(amount_specified, pool_info.quote_reserve, pool_info.base_reserve);
                let max_quote_amount_in = max_amount_with_slippage(amount_specified, slippage_bps as u64);
                
                // Verify amount doesn't exceed pool reserves
                if base_amount_out > pool_info.base_reserve {
                    return Err(anyhow!("Cannot buy more base tokens than the pool reserves"));
                }
                
                // Create buy accounts
                (
                    base_amount_out,
                    max_quote_amount_in,
                    create_buy_accounts(
                        pool_info.pool_id,
                        owner,
                        mint,
                        sol_mint,
                        out_ata,
                        in_ata,
                        pool_info.pool_base_account,
                        pool_info.pool_quote_account,
                        pool_info.coin_creator,
                    )?,
                )
            },
            SwapDirection::Sell => {
                // Verify token account exists using our global tracker
                if !in_ata_exists {
                    return Err(anyhow!("Token ATA does not exist in our wallet, cannot sell"));
                }
                
                // Get token account info and mint info in parallel - for sells we still need
                // to check balance and decimals
                let (in_account, in_mint) = if let Some(nonblocking_client) = &self.rpc_nonblocking_client {
                    let account_future = crate::core::token::get_account_info(
                        nonblocking_client.clone(),
                        token_in,
                        in_ata,
                    );
                    
                    let mint_future = crate::core::token::get_mint_info(
                        nonblocking_client.clone(),
                        self.keypair.clone(),
                        token_in,
                    );
                    
                    // Run both futures concurrently
                    let (account_result, mint_result) = tokio::join!(account_future, mint_future);
                    
                    (account_result?, mint_result?)
                } else {
                    return Err(anyhow!("RPC nonblocking client not initialized"));
                };
                
                // Calculate amount to sell
                let amount = match swap_config.in_type {
                    SwapInType::Qty => {
                        ui_amount_to_amount(swap_config.amount_in, in_mint.base.decimals)
                    },
                    SwapInType::Pct => {
                        let amount_in_pct = swap_config.amount_in.min(1.0);
                        if amount_in_pct == 1.0 {
                            // Sell all tokens and close account
                            close_instruction = Some(spl_token::instruction::close_account(
                                &Pubkey::from_str(TOKEN_PROGRAM)?,
                                &in_ata,
                                &owner,
                                &owner,
                                &[&owner],
                            )?);
                            in_account.base.amount
                        } else {
                            (amount_in_pct * 100.0) as u64 * in_account.base.amount / 100
                        }
                    }
                };
                
                // Validate amount
                if amount == 0 {
                    return Err(anyhow!("Amount is zero, cannot sell"));
                }
                
                if amount > in_account.base.amount {
                    return Err(anyhow!("Sell amount exceeds account balance"));
                }
                
                // Calculate expected SOL output
                let base_amount_in = amount;
                let quote_amount_out = calculate_sell_quote_amount(base_amount_in, pool_info.base_reserve, pool_info.quote_reserve);
                let min_quote_amount_out = min_amount_with_slippage(quote_amount_out, slippage_bps as u64);
                
                // Create sell accounts
                (
                    base_amount_in,
                    min_quote_amount_out,
                    create_sell_accounts(
                        pool_info.pool_id,
                        owner,
                        mint,
                        sol_mint,
                        in_ata,
                        out_ata,
                        pool_info.pool_base_account,
                        pool_info.pool_quote_account,
                        pool_info.coin_creator,
                    )?,
                )
            }
        };
        
        
        // Create the swap instruction
        let swap_instruction = create_swap_instruction(
            pump_program,
            discriminator,
            base_amount,
            quote_amount,
            accounts,
        );
        
        // Build the final transaction instructions
        let mut instructions = vec![];
        
        // Add token account creation if needed
        if let Some(create_instruction) = create_instruction {
            logger.log("Adding token account creation instruction to transaction".to_string());
            instructions.push(create_instruction);
        }
        
        // Add the swap instruction
        if base_amount > 0 {
            logger.log("Adding swap instruction to transaction".to_string());
            instructions.push(swap_instruction);
        }
        
        // Add close instructions if needed
        if let Some(close_instruction) = close_instruction {
            logger.log("Adding close instructions to transaction".to_string());
            instructions.push(close_instruction);
        }
        // Validate we have instructions
        if instructions.is_empty() {
            return Err(anyhow!("Instructions is empty, no txn required."));
        }
        
        logger.log(format!("Built transaction with {} instructions", instructions.len()));
        
        // Return instructions, token price, and blockhash
        Ok((self.keypair.clone(), instructions, token_price, blockhash))
    }

    /// Get or create token account for a given mint
    pub async fn get_or_create_token_account(&self, owner: Pubkey, mint: Pubkey) -> Result<Pubkey, String> {
        // Find the associated token account
        let token_account = get_associated_token_address(&owner, &mint);
        
        // Check if account exists
        let exists = match self.rpc_nonblocking_client.as_ref() {
            Some(client) => {
                match client.get_account(&token_account).await {
                    Ok(_) => true, // Account exists if we got an Ok result
                    Err(_) => false, // Account doesn't exist or there was an error
                }
            },
            None => false,
        };
        
        if !exists {
            // Create account instruction would be needed here
            // This would typically be added to the transaction
            // For now, just return the ATA address and let the caller handle creation
        }
        
        Ok(token_account)
    }
    
    /// Get token balance for an account
    pub async fn get_token_balance(&self, token_account: Pubkey) -> Result<f64, String> {
        match self.rpc_nonblocking_client.as_ref() {
            Some(client) => {
                match client.get_token_account_balance(&token_account).await {
                    Ok(balance) => {
                        Ok(balance.ui_amount.unwrap_or(0.0))
                    },
                    Err(e) => Err(format!("Failed to get token balance: {}", e)),
                }
            },
            None => Err("RPC client not initialized".to_string()),
        }
    }

    /// Get token price in SOL - optimized with caching
    pub async fn get_token_price(&self, token_mint: &str) -> Result<f64, String> { 
        let logger = Logger::new("[PUMP-SWAP-PRICE] => ".yellow().to_string());
        logger.log(format!("Getting price for token: {}", token_mint));
        
        // Parse mint pubkey
        let mint = match Pubkey::from_str(token_mint) {
            Ok(pubkey) => pubkey,
            Err(e) => return Err(format!("Invalid token mint: {}", e)),
        };
        
        // Function to calculate price from pool reserves
        let calculate_price = |base_reserve: u64, quote_reserve: u64| -> f64 {
            if base_reserve == 0 {
                return 0.0;
            }
            
            // SOL has 9 decimals
            const SOL_DECIMALS: i32 = 9;
            // Most tokens use 9 decimals, but we could fetch the actual value if needed
            const TOKEN_DECIMALS: i32 = 9;
            
            // Convert to floating point and adjust for decimals in a single step
            let decimal_adjustment = 10f64.powi(SOL_DECIMALS - TOKEN_DECIMALS);
            (quote_reserve as f64) / (base_reserve as f64) * decimal_adjustment
        };
        
        // Check cache first for quick response
        if let Some(cached_pool) = POOL_CACHE.get(&mint) {
            logger.log("Using cached pool data for price calculation".to_string());
            
            // Calculate price from cached data
            let price = calculate_price(cached_pool.base_reserve, cached_pool.quote_reserve);
            
            logger.log(format!("Calculated price from cache: {} SOL/token", price));
            return Ok(price);
        }
        
        // Get RPC client if not using cached data
        let rpc_client = match &self.rpc_client {
            Some(client) => client,
            None => return Err("RPC client not initialized".to_string()),
        };
        
        // Get pool info for the token
        match get_pool_info(rpc_client.clone(), mint).await {
            Ok(pool_info) => {
                // Calculate price using the helper function
                let price = calculate_price(pool_info.base_reserve, pool_info.quote_reserve);
                
                logger.log(format!("Calculated price: {} SOL/token", price));
                Ok(price)
            },
            Err(e) => Err(format!("Failed to get pool info: {}", e)),
        }
    }

    /// Create a buy instruction for PumpSwap
    pub async fn create_buy_instruction(
        &self,
        token_mint: Pubkey,
        token_account: Pubkey,
        sol_amount: f64,
        slippage_bps: u64,
    ) -> Result<Instruction, String> {
        let logger = Logger::new("[CREATE-BUY-IX] => ".green().to_string());
        logger.log(format!("Creating buy instruction for {} SOL", sol_amount));
        
        // Get owner and SOL mint
        let owner = self.keypair.pubkey();
        let sol_mint = Pubkey::from_str(SOL_MINT).map_err(|e| format!("Invalid SOL mint: {}", e))?;
        
        // Get WSOL account
        let wsol_account = get_associated_token_address(&owner, &sol_mint);
        
        // Get or create pool info
        let pool_info = match self.rpc_client.as_ref() {
            Some(client) => {
                match get_pool_info(client.clone(), token_mint).await {
                    Ok(info) => info,
                    Err(e) => return Err(format!("Failed to get pool info: {}", e)),
                }
            },
            None => return Err("RPC client not initialized".to_string()),
        };
        
        // Convert SOL amount to lamports
        let sol_lamports = ui_amount_to_amount(sol_amount, 9); // SOL has 9 decimals
        
        // Calculate expected token amount out
        let base_amount_out = calculate_buy_base_amount(
            sol_lamports, 
            pool_info.quote_reserve, 
            pool_info.base_reserve
        );
        
        // Calculate maximum SOL to spend with slippage
        let max_sol_in = max_amount_with_slippage(sol_lamports, slippage_bps);
        
        // Check pool constraints
        if base_amount_out == 0 {
            return Err("Calculated token amount is zero - check pool liquidity".to_string());
        }
        
        if base_amount_out > pool_info.base_reserve {
            return Err("Buy amount exceeds pool reserves".to_string());
        }
        
        // Get PumpSwap program pubkey
        let pump_program = Pubkey::from_str(PUMP_SWAP_PROGRAM)
            .map_err(|e| format!("Invalid program ID: {}", e))?;
        
        // Create the buy accounts
        let accounts = match create_buy_accounts(
            pool_info.pool_id,
            owner,
            token_mint,
            sol_mint,
            token_account,
            wsol_account,
            pool_info.pool_base_account,
            pool_info.pool_quote_account,
            pool_info.coin_creator,
        ) {
            Ok(accounts) => accounts,
            Err(e) => return Err(format!("Failed to create buy accounts: {}", e)),
        };
        
        // Create the swap instruction
        let instruction = create_swap_instruction(
            pump_program,
            BUY_DISCRIMINATOR,
            base_amount_out,
            max_sol_in,
            accounts,
        );
        
        logger.log(format!(
            "Created buy instruction: {} SOL for {} tokens",
            amount_to_ui_amount(max_sol_in, 9),
            amount_to_ui_amount(base_amount_out, 9) // Assuming token decimals is 9 for simplicity
        ));
        
        Ok(instruction)
    }
    
    /// Create a sell instruction for PumpSwap
    pub async fn create_sell_instruction(
        &self,
        token_mint: Pubkey,
        token_account: Pubkey,
        token_amount: f64,
        slippage_bps: u64,
    ) -> Result<Instruction, String> {
        let logger = Logger::new("[CREATE-SELL-IX] => ".green().to_string());
        logger.log(format!("Creating sell instruction for {} tokens", token_amount));
        
        // Get owner and SOL mint
        let owner = self.keypair.pubkey();
        let sol_mint = Pubkey::from_str(SOL_MINT).map_err(|e| format!("Invalid SOL mint: {}", e))?;
        
        // Get WSOL account
        let wsol_account = get_associated_token_address(&owner, &sol_mint);
        
        // Get pool info
        let pool_info = match self.rpc_client.as_ref() {
            Some(client) => {
                match get_pool_info(client.clone(), token_mint).await {
                    Ok(info) => info,
                    Err(e) => return Err(format!("Failed to get pool info: {}", e)),
                }
            },
            None => return Err("RPC client not initialized".to_string()),
        };
        
        // Get token decimals from mint info
        let token_decimals = match self.rpc_nonblocking_client.as_ref() {
            Some(client) => {
                match token::get_mint_info(client.clone(), self.keypair.clone(), token_mint).await {
                    Ok(mint_info) => mint_info.base.decimals,
                    Err(_) => 9, // Default to 9 if we can't get the decimals
                }
            },
            None => 9, // Default to 9 if no client available
        };
        
        // Convert token amount to raw amount
        let token_raw_amount = ui_amount_to_amount(token_amount, token_decimals);
        
        // Calculate expected SOL output
        let quote_amount_out = calculate_sell_quote_amount(
            token_raw_amount, 
            pool_info.base_reserve, 
            pool_info.quote_reserve
        );
        
        // Apply slippage tolerance to minimum SOL output
        let min_sol_out = min_amount_with_slippage(quote_amount_out, slippage_bps);
        
        // Check pool constraints
        if quote_amount_out == 0 {
            return Err("Calculated SOL amount is zero - check pool liquidity".to_string());
        }
        
        if quote_amount_out > pool_info.quote_reserve {
            return Err("Sell would drain too much SOL from pool".to_string());
        }
        
        // Get PumpSwap program pubkey
        let pump_program = Pubkey::from_str(PUMP_SWAP_PROGRAM)
            .map_err(|e| format!("Invalid program ID: {}", e))?;
        
        // Create the sell accounts
        let accounts = match create_sell_accounts(
            pool_info.pool_id,
            owner,
            token_mint,
            sol_mint,
            token_account,
            wsol_account,
            pool_info.pool_base_account,
            pool_info.pool_quote_account,
            pool_info.coin_creator,
        ) {
            Ok(accounts) => accounts,
            Err(e) => return Err(format!("Failed to create sell accounts: {}", e)),
        };
        
        // Create the swap instruction
        let instruction = create_swap_instruction(
            pump_program,
            SELL_DISCRIMINATOR,
            token_raw_amount,
            min_sol_out,
            accounts,
        );
        
        logger.log(format!(
            "Created sell instruction: {} tokens for min {} SOL",
            token_amount,
            amount_to_ui_amount(min_sol_out, 9)
        ));
        
        Ok(instruction)
    }
}

/// Get the PumpSwap pool information for a specific token mint
pub async fn get_pool_info(
    rpc_client: Arc<anchor_client::solana_client::rpc_client::RpcClient>,
    mint: Pubkey,
) -> Result<PumpSwapPool> {
    let logger = Logger::new("[PUMPSWAP-GET-POOL-INFO] => ".blue().to_string());
    
    // Check cache first
    if let Some(pool) = POOL_CACHE.get(&mint) {
        logger.log(format!("Using cached pool info for mint {}", mint));
        return Ok(pool);
    }
    
    // Initialize
    let sol_mint = Pubkey::from_str(SOL_MINT)?;
    let pump_program = Pubkey::from_str(PUMP_SWAP_PROGRAM)?;
    
    // Use getProgramAccounts with config for better efficiency
    let mut pool_id = Pubkey::default();
    let mut retry_count = 0;
    let max_retries = 2;
    
    // Try to find the pool
    while retry_count < max_retries && pool_id == Pubkey::default() {
        match rpc_client.get_program_accounts_with_config(
            &pump_program,
            RpcProgramAccountsConfig {
                filters: Some(vec![
                    RpcFilterType::DataSize(300),
                    solana_client::rpc_filter::RpcFilterType::Memcmp(Memcmp::new(43, MemcmpEncodedBytes::Base64(base64::encode(mint.to_bytes())))),
                ]),
                account_config: RpcAccountInfoConfig {
                    encoding: Some(UiAccountEncoding::Base64),
                    ..RpcAccountInfoConfig::default()
                },
                ..RpcProgramAccountsConfig::default()
            },
        ) {
            Ok(accounts) => {
                // Loop through accounts to find one with our mint
                for (pubkey, account) in accounts.iter() {
                    if account.data.len() >= 75 {
                        if let Ok(pubkey_from_data) = Pubkey::try_from(&account.data[43..75]) {
                            if pubkey_from_data == mint {
                                pool_id = *pubkey;
                                break;
                            }
                        }
                    }
                }
                
                if pool_id != Pubkey::default() {
                    break; // Pool found, exit the retry loop
                } else if retry_count + 1 < max_retries {
                    logger.log("No pools found for the given mint, retrying...".to_string());
                }
            }
            Err(err) => {
                logger.log(format!("Error getting program accounts (attempt {}/{}): {}", 
                                  retry_count + 1, max_retries, err));
            }
        }
        
        retry_count += 1;
        if retry_count < max_retries {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    
    if pool_id == Pubkey::default() {
        return Err(anyhow!("Failed to find PumpSwap pool for mint {}", mint));
    }
    
    // Derive token accounts
    let pool_base_account = get_associated_token_address(&pool_id, &mint);
    let pool_quote_account = get_associated_token_address(&pool_id, &sol_mint);
    
    // Get token balances with a single batch call
    let accounts_to_check = vec![pool_base_account, pool_quote_account];
    
    // Get multiple accounts in a single RPC call
    let accounts = match rpc_client.get_multiple_accounts(&accounts_to_check) {
        Ok(accounts) => accounts,
        Err(e) => {
            logger.log(format!("Warning: Failed to get pool accounts in batch: {}", e));
            // Use default values instead of failing
            vec![None, None]
        }
    };
    
    // Extract balances with proper error handling
    let base_balance = if let Some(account_data) = &accounts[0] {
        match spl_token::state::Account::unpack(&account_data.data) {
            Ok(token_account) => token_account.amount,
            Err(e) => {
                logger.log(format!("Warning: Failed to unpack base token account: {}", e));
                10_000_000_000_000 // Fallback value
            }
        }
    } else {
        logger.log("Warning: Base token account not found".to_string());
        10_000_000_000_000 // Fallback value
    };
    
    let quote_balance = if let Some(account_data) = &accounts[1] {
        match spl_token::state::Account::unpack(&account_data.data) {
            Ok(token_account) => token_account.amount,
            Err(e) => {
                logger.log(format!("Warning: Failed to unpack quote token account: {}", e));
                10_000_000_000 // Fallback value (10 SOL)
            }
        }
    } else {
        logger.log("Warning: Quote token account not found".to_string());
        10_000_000_000 // Fallback value (10 SOL)
    };
    
    // Create pool info object
    let pool_info = PumpSwapPool {
        pool_id,
        base_mint: mint,
        quote_mint: sol_mint,
        lp_mint: Pubkey::default(), // Not used for swaps
        pool_base_account,
        pool_quote_account,
        base_reserve: base_balance,
        quote_reserve: quote_balance,
        coin_creator: Pubkey::default(), // Default value
    };
    
    // Cache the result for future use
    POOL_CACHE.insert(mint, pool_info.clone(), None);
    
    Ok(pool_info)
}

/// Helper function to get token balance with retries
async fn get_token_balance_with_retry(
    rpc_client: &Arc<anchor_client::solana_client::rpc_client::RpcClient>,
    account: &Pubkey,
    max_retries: u8,
) -> Result<u64> {
    let mut retry_count = 0;
    let logger = Logger::new("[GET-TOKEN-BALANCE] => ".cyan().to_string());
    
    while retry_count < max_retries {
        match rpc_client.get_token_account_balance(account) {
            Ok(balance) => {
                return Ok(match balance.ui_amount {
                    Some(amount) => (amount * (10f64.powf(balance.decimals as f64))) as u64,
                    None => 0,
                });
            },
            Err(e) => {
                retry_count += 1;
                if retry_count >= max_retries {
                    return Err(anyhow!("Failed to get token balance after {} attempts: {}", max_retries, e));
                }
                logger.log(format!("Error getting token balance (attempt {}/{}): {}", 
                                  retry_count, max_retries, e));
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
    
    Err(anyhow!("Failed to get token balance after max retries"))
}

/// Calculate the amount of base tokens received for a given quote amount in buy operation
fn calculate_buy_base_amount(quote_amount_in: u64, quote_reserve: u64, base_reserve: u64) -> u64 {
    // For buys in constant product AMM:
    // quote_reserve * base_reserve = (quote_reserve + quote_amount_in) * (base_reserve - base_amount_out)
    // Solving for base_amount_out:
    // base_amount_out = base_reserve - (quote_reserve * base_reserve) / (quote_reserve + quote_amount_in)
    
    if quote_amount_in == 0 || base_reserve == 0 || quote_reserve == 0 {
        return 0;
    }
    
    let quote_reserve_after = quote_reserve.checked_add(quote_amount_in).unwrap_or(quote_reserve);
    let numerator = (quote_reserve as u128).checked_mul(base_reserve as u128).unwrap_or(0);
    let denominator = quote_reserve_after as u128;
    
    if denominator == 0 {
        return 0;
    }
    
    let base_reserve_after = numerator.checked_div(denominator).unwrap_or(0);
    let base_amount_out = base_reserve.checked_sub(base_reserve_after as u64).unwrap_or(0);
    
    base_amount_out
}

/// Calculate the amount of quote tokens received for a given base amount in sell operation
fn calculate_sell_quote_amount(base_amount_in: u64, base_reserve: u64, quote_reserve: u64) -> u64 {
    // For sells in constant product AMM:
    // quote_reserve * base_reserve = (quote_reserve - quote_amount_out) * (base_reserve + base_amount_in)
    // Solving for quote_amount_out:
    // quote_amount_out = quote_reserve - (quote_reserve * base_reserve) / (base_reserve + base_amount_in)
    
    if base_amount_in == 0 || base_reserve == 0 || quote_reserve == 0 {
        return 0;
    }
    
    let base_reserve_after = base_reserve.checked_add(base_amount_in).unwrap_or(base_reserve);
    let numerator = (quote_reserve as u128).checked_mul(base_reserve as u128).unwrap_or(0);
    let denominator = base_reserve_after as u128;
    
    if denominator == 0 {
        return 0;
    }
    
    let quote_reserve_after = numerator.checked_div(denominator).unwrap_or(0);
    let quote_amount_out = quote_reserve.checked_sub(quote_reserve_after as u64).unwrap_or(0);
    
    quote_amount_out
}

/// Calculate the minimum amount with slippage tolerance
fn min_amount_with_slippage(input_amount: u64, slippage_bps: u64) -> u64 {
    input_amount
        .checked_mul(TEN_THOUSAND.checked_sub(slippage_bps).unwrap_or(TEN_THOUSAND))
        .unwrap_or(input_amount)
        .checked_div(TEN_THOUSAND)
        .unwrap_or(input_amount)
}

/// Calculate the maximum amount with slippage tolerance
fn max_amount_with_slippage(input_amount: u64, slippage_bps: u64) -> u64 {
    input_amount
        .checked_mul(slippage_bps.checked_add(TEN_THOUSAND).unwrap_or(TEN_THOUSAND))
        .unwrap_or(input_amount)
        .checked_div(TEN_THOUSAND)
        .unwrap_or(input_amount)
}

/// Create accounts for buy operation
fn create_buy_accounts(
    pool_id: Pubkey,
    user: Pubkey,
    base_mint: Pubkey,
    quote_mint: Pubkey,
    user_base_token_account: Pubkey,
    wsol_account: Pubkey,
    pool_base_token_account: Pubkey,
    pool_quote_token_account: Pubkey,
    coin_creator: Pubkey,
) -> Result<Vec<AccountMeta>> {
    let global_config = Pubkey::from_str(PUMP_GLOBAL_CONFIG)?;
    let fee_recipient = Pubkey::from_str(PUMP_FEE_RECIPIENT)?;
    let fee_recipient_ata = get_associated_token_address(&fee_recipient, &quote_mint);
    let event_authority = Pubkey::from_str(PUMP_EVENT_AUTHORITY)?;
    let pump_program = Pubkey::from_str(PUMP_SWAP_PROGRAM)?;
    let token_program = Pubkey::from_str(TOKEN_PROGRAM)?;
    let associated_token_program = Pubkey::from_str(ASSOCIATED_TOKEN_PROGRAM)?;
    let (coin_creator_vault_authority, _) = Pubkey::find_program_address(
        &[b"creator_vault", coin_creator.as_ref()],
        &pump_program,
    );
    let coin_creator_vault_ata = get_associated_token_address(&coin_creator_vault_authority, &quote_mint);
    
    Ok(vec![
        AccountMeta::new_readonly(pool_id, false),
        AccountMeta::new(user, true),
        AccountMeta::new_readonly(global_config, false),
        AccountMeta::new_readonly(base_mint, false),
        AccountMeta::new_readonly(quote_mint, false),
        AccountMeta::new(user_base_token_account, false),
        AccountMeta::new(wsol_account, false),
        AccountMeta::new(pool_base_token_account, false),
        AccountMeta::new(pool_quote_token_account, false),
        AccountMeta::new_readonly(fee_recipient, false),
        AccountMeta::new(fee_recipient_ata, false),
        AccountMeta::new_readonly(token_program, false),
        AccountMeta::new_readonly(token_program, false),
        AccountMeta::new_readonly(system_program::id(), false),
        AccountMeta::new_readonly(associated_token_program, false),
        AccountMeta::new_readonly(event_authority, false),
        AccountMeta::new_readonly(pump_program, false),
        AccountMeta::new(coin_creator_vault_ata, false),
        AccountMeta::new_readonly(coin_creator_vault_authority, false),
        ])
}

/// Create accounts for sell operation
fn create_sell_accounts(
    pool_id: Pubkey,
    user: Pubkey,
    base_mint: Pubkey,
    quote_mint: Pubkey,
    user_base_token_account: Pubkey,
    wsol_account: Pubkey,
    pool_base_token_account: Pubkey,
    pool_quote_token_account: Pubkey,
    coin_creator: Pubkey,
) -> Result<Vec<AccountMeta>> {
    let global_config = Pubkey::from_str(PUMP_GLOBAL_CONFIG)?;
    let fee_recipient = Pubkey::from_str(PUMP_FEE_RECIPIENT)?;
    let fee_recipient_ata = get_associated_token_address(&fee_recipient, &quote_mint);
    let event_authority = Pubkey::from_str(PUMP_EVENT_AUTHORITY)?;
    let pump_program = Pubkey::from_str(PUMP_SWAP_PROGRAM)?;
    let token_program = Pubkey::from_str(TOKEN_PROGRAM)?;
    let associated_token_program = Pubkey::from_str(ASSOCIATED_TOKEN_PROGRAM)?;
    println!("coin_creator: {}", coin_creator); 
    let (coin_creator_vault_authority, _) = Pubkey::find_program_address(
        &[b"creator_vault", coin_creator.as_ref()],
        &pump_program,
    );
    let coin_creator_vault_ata = get_associated_token_address(&coin_creator_vault_authority, &quote_mint);

    Ok(vec![
        AccountMeta::new_readonly(pool_id, false),
        AccountMeta::new(user, true),
        AccountMeta::new_readonly(global_config, false),
        AccountMeta::new_readonly(base_mint, false),
        AccountMeta::new_readonly(quote_mint, false),
        AccountMeta::new(user_base_token_account, false),
         AccountMeta::new(wsol_account, false),
         AccountMeta::new(pool_base_token_account, false),
         AccountMeta::new(pool_quote_token_account, false),
         AccountMeta::new_readonly(fee_recipient, false),
         AccountMeta::new(fee_recipient_ata, false),
         AccountMeta::new_readonly(token_program, false),
         AccountMeta::new_readonly(token_program, false),
         AccountMeta::new_readonly(system_program::id(), false),
         AccountMeta::new_readonly(associated_token_program, false),
         AccountMeta::new_readonly(event_authority, false),
         AccountMeta::new_readonly(pump_program, false),
         AccountMeta::new(coin_creator_vault_ata, false),
         AccountMeta::new_readonly(coin_creator_vault_authority, false),
])
}

/// Create a swap instruction with the given parameters
fn create_swap_instruction(
    program_id: Pubkey,
    discriminator: [u8; 8],
    base_amount: u64,
    quote_amount: u64,
    accounts: Vec<AccountMeta>,
) -> Instruction {
    // Create the data buffer: discriminator + base_amount + quote_amount
    let mut data = Vec::with_capacity(24); // 8 + 8 + 8 bytes
    data.extend_from_slice(&discriminator);
    data.extend_from_slice(&base_amount.to_le_bytes());
    data.extend_from_slice(&quote_amount.to_le_bytes());
    
    Instruction {
        program_id,
        accounts,
        data,
    }
}

