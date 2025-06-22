use std::{str::FromStr, sync::Arc, time::Duration};

use anyhow::{anyhow, Result};
use borsh::from_slice;
use borsh_derive::{BorshDeserialize, BorshSerialize};
use chrono::Utc;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use anchor_client::solana_sdk::{
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    system_program,
};
use spl_associated_token_account::{
    get_associated_token_address, instruction::create_associated_token_account,
};
use spl_token::{amount_to_ui_amount, ui_amount_to_amount};
use spl_token_client::token::TokenError;
use tokio::time::Instant;

use crate::{
    common::{config::SwapConfig, logger::Logger},
    core::token,
    engine::{monitor::BondingCurveInfo, swap::{SwapDirection, SwapInType}},
};

pub const TEN_THOUSAND: u64 = 10000;
pub const TOKEN_PROGRAM: &str = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA";
pub const RENT_PROGRAM: &str = "SysvarRent111111111111111111111111111111111";
pub const ASSOCIATED_TOKEN_PROGRAM: &str = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL";
pub const PUMP_GLOBAL: &str = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf";
pub const PUMP_FEE_RECIPIENT: &str = "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM";
pub const PUMP_PROGRAM: &str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P";
// pub const PUMP_FUN_MINT_AUTHORITY: &str = "TSLvdd1pWpHVjahSpsvCXUbgwsL3JAcvokwaKt1eokM";
pub const PUMP_ACCOUNT: &str = "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1";
pub const PUMP_BUY_METHOD: u64 = 16927863322537952870;
pub const PUMP_SELL_METHOD: u64 = 12502976635542562355;
pub const PUMP_FUN_CREATE_IX_DISCRIMINATOR: &[u8] = &[24, 30, 200, 40, 5, 28, 7, 119];
pub const INITIAL_VIRTUAL_SOL_RESERVES: u64 = 30_000_000_000;
pub const INITIAL_VIRTUAL_TOKEN_RESERVES: u64 = 1_073_000_000_000_000;
pub const TOKEN_TOTAL_SUPPLY: u64 = 1_000_000_000_000_000;

#[derive(Clone)]
pub struct Pump {
    pub rpc_nonblocking_client: Arc<anchor_client::solana_client::nonblocking::rpc_client::RpcClient>,
    pub keypair: Arc<Keypair>,
    pub rpc_client: Option<Arc<anchor_client::solana_client::rpc_client::RpcClient>>,
}

impl Pump {
    pub fn new(
        rpc_nonblocking_client: Arc<anchor_client::solana_client::nonblocking::rpc_client::RpcClient>,
        rpc_client: Arc<anchor_client::solana_client::rpc_client::RpcClient>,
        keypair: Arc<Keypair>,
    ) -> Self {
        Self {
            rpc_nonblocking_client,
            keypair,
            rpc_client: Some(rpc_client),
        }
    }

    pub async fn build_swap_ixn_by_mint(
        &self,
        mint_str: &str,
        bonding_curve_info: Option<BondingCurveInfo>,
        swap_config: SwapConfig,
        start_time: Instant,
    ) -> Result<(Arc<Keypair>, Vec<Instruction>, f64)> {
        let logger = Logger::new("[PUMPFUN-SWAP-BY-MINT] => ".blue().to_string());
        // logger.log(
        //     format!(
        //         "[SWAP({:?})-BEGIN]({}) - {} :: {:?}",
        //         swap_config.swap_direction,
        //         mint_str,
        //         chrono::Utc::now(),
        //         start_time.elapsed()
        //     )
        //     .yellow()
        //     .to_string(),
        // );
        // Constants
        // ---------------------------------------------------
        let slippage_bps = swap_config.slippage * 100;
        let owner = self.keypair.pubkey();
        let mint = Pubkey::from_str(mint_str).map_err(|_| anyhow!(""))?;
        let program_id = spl_token::ID;
        let native_mint = spl_token::native_mint::ID;
        let (token_in, token_out, pump_method) = match swap_config.swap_direction {
            SwapDirection::Buy => (native_mint, mint, PUMP_BUY_METHOD),
            SwapDirection::Sell => (mint, native_mint, PUMP_SELL_METHOD),
        };
        let pump_program = Pubkey::from_str(PUMP_PROGRAM)?;

        let mut create_instruction = None;
        let mut close_instruction = None;

        // RPC requests
        // ---------------------------------------------------
        // let nonblocking_clinet_clone = self.rpc_nonblocking_client.clone();
        // let bonding_curve_handle = tokio::spawn(get_bonding_curve_account(
        //     self.rpc_client.clone().unwrap(),
        //     mint,
        //     pump_program,
        // ));
        // let blockhash_handle =
        //     tokio::spawn(async move { nonblocking_clinet_clone.get_latest_blockhash().await });
        
        let (bonding_curve, associated_bonding_curve, bonding_curve_reserves) =
            if let Some(bonding_curve_info) = bonding_curve_info {
                get_bonding_curve_account_by_calc(bonding_curve_info, mint)
            } else {
                get_bonding_curve_account(self.rpc_client.clone().unwrap(), mint, pump_program).await?
            };

        // Calculate tokens out
        let virtual_sol_reserves = bonding_curve_reserves.virtual_sol_reserves as u128;
        let virtual_token_reserves = bonding_curve_reserves.virtual_token_reserves as u128;

        let in_ata = token::get_associated_token_address(
            self.rpc_nonblocking_client.clone(),
            self.keypair.clone(),
            &token_in,
            &owner,
        );

        let out_ata = token::get_associated_token_address(
            self.rpc_nonblocking_client.clone(),
            self.keypair.clone(),
            &token_out,
            &owner,
        );

        let (amount_specified, _amount_ui_pretty) = match swap_config.swap_direction {
            SwapDirection::Buy => {
                // Create base ATA if it doesn't exist.
                // ----------------------------
                match token::get_account_info(
                    self.rpc_nonblocking_client.clone(),
                    token_out,
                    out_ata,
                )
                .await
                {
                    Ok(_) => {
                        // Base ata exists. skipping creation..
                        // --------------------------
                    }
                    Err(TokenError::AccountNotFound) | Err(TokenError::AccountInvalidOwner) => {
                        // "Base ATA for mint {} does not exist. will be create", token_out
                        // --------------------------
                        create_instruction = Some(create_associated_token_account(
                            &owner,
                            &owner,
                            &token_out,
                            &program_id,
                        ));
                    }
                    Err(_) => {
                        // Error retrieving out ATA
                        // ---------------------------
                    }
                }
                (
                    ui_amount_to_amount(swap_config.amount_in, spl_token::native_mint::DECIMALS),
                    (swap_config.amount_in, spl_token::native_mint::DECIMALS),
                )
            }
            SwapDirection::Sell => {
                let in_account_handle = tokio::spawn(token::get_account_info(
                    self.rpc_nonblocking_client.clone(),
                    token_in,
                    in_ata,
                ));
                let in_mint_handle = tokio::spawn(token::get_mint_info(
                    self.rpc_nonblocking_client.clone(),
                    self.keypair.clone(),
                    token_in,
                ));
                let (in_account, in_mint) =
                    match tokio::try_join!(in_account_handle, in_mint_handle) {
                        Ok((in_account_result, in_mint_result)) => {
                            let in_account_result = in_account_result?;
                            let in_mint_result = in_mint_result?;
                            (in_account_result, in_mint_result)
                        }
                        Err(err) => {
                            logger.log(format!("Failed with {}, ", err).red().to_string());
                            return Err(anyhow!(format!("{}", err)));
                        }
                    };
                let amount = match swap_config.in_type {
                    SwapInType::Qty => {
                        ui_amount_to_amount(swap_config.amount_in, in_mint.base.decimals)
                    }
                    SwapInType::Pct => {
                        let amount_in_pct = swap_config.amount_in.min(1.0);
                        if amount_in_pct == 1.0 {
                            // Sell all. will be close ATA for mint {token_in}
                            // --------------------------------
                            close_instruction = Some(spl_token::instruction::close_account(
                                &program_id,
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
                (
                    amount,
                    (
                        amount_to_ui_amount(amount, in_mint.base.decimals),
                        in_mint.base.decimals,
                    ),
                )
            }
        };

        let token_price: f64 = (virtual_sol_reserves as f64) / (virtual_token_reserves as f64);

        let (token_amount, sol_amount_threshold, input_accouts) = match swap_config.swap_direction {
            SwapDirection::Buy => {
                let max_sol_cost = max_amount_with_slippage(amount_specified, slippage_bps);
                let amount_result = u128::from(amount_specified)
                    .checked_mul(virtual_token_reserves)
                    .expect("Failed to multiply amount_specified by virtual_token_reserves: overflow occurred.")
                    .checked_div(virtual_sol_reserves)
                    .expect("Failed to divide the result by virtual_sol_reserves: division by zero or overflow occurred.");
                (
                    amount_result as u64,
                    max_sol_cost,
                    vec![
                        AccountMeta::new_readonly(Pubkey::from_str(PUMP_GLOBAL)?, false),
                        AccountMeta::new(Pubkey::from_str(PUMP_FEE_RECIPIENT)?, false),
                        AccountMeta::new_readonly(mint, false),
                        AccountMeta::new(bonding_curve, false),
                        AccountMeta::new(associated_bonding_curve, false),
                        AccountMeta::new(out_ata, false),
                        AccountMeta::new(owner, true),
                        AccountMeta::new_readonly(system_program::id(), false),
                        AccountMeta::new_readonly(program_id, false),
                        AccountMeta::new_readonly(Pubkey::from_str(RENT_PROGRAM)?, false),
                        AccountMeta::new_readonly(Pubkey::from_str(PUMP_ACCOUNT)?, false),
                        AccountMeta::new_readonly(pump_program, false),
                    ],
                )
            }
            SwapDirection::Sell => {
                let sol_output = u128::from(amount_specified)
                .checked_mul(virtual_sol_reserves)
                .expect("Failed to multiply amount_specified by virtual_sol_reserves: overflow occurred.")
                .checked_div(virtual_token_reserves)
                .expect("Failed to divide the result by virtual_token_reserves: division by zero or overflow occurred.");
                let min_sol_output = min_amount_with_slippage(sol_output as u64, slippage_bps);

                (
                    amount_specified,
                    min_sol_output,
                    vec![
                        AccountMeta::new_readonly(Pubkey::from_str(PUMP_GLOBAL)?, false),
                        AccountMeta::new(Pubkey::from_str(PUMP_FEE_RECIPIENT)?, false),
                        AccountMeta::new_readonly(mint, false),
                        AccountMeta::new(bonding_curve, false),
                        AccountMeta::new(associated_bonding_curve, false),
                        AccountMeta::new(in_ata, false),
                        AccountMeta::new(owner, true),
                        AccountMeta::new_readonly(system_program::id(), false),
                        AccountMeta::new_readonly(
                            Pubkey::from_str(ASSOCIATED_TOKEN_PROGRAM)?,
                            false,
                        ),
                        AccountMeta::new_readonly(program_id, false),
                        AccountMeta::new_readonly(Pubkey::from_str(PUMP_ACCOUNT)?, false),
                        AccountMeta::new_readonly(pump_program, false),
                    ],
                )
            }
        };

        // Constants-Instruction Configuration
        // -------------------
        let build_swap_instruction = Instruction::new_with_bincode(
            pump_program,
            &(pump_method, token_amount, sol_amount_threshold),
            input_accouts,
        );

        let mut instructions = vec![];
        if let Some(create_instruction) = create_instruction {
            instructions.push(create_instruction);
        }
        if amount_specified > 0 {
            instructions.push(build_swap_instruction)
        }
        if let Some(close_instruction) = close_instruction {
            instructions.push(close_instruction);
        }
        if instructions.is_empty() {
            return Err(anyhow!("Instructions is empty, no txn required."
                .red()
                .italic()
                .to_string()));
        }
        // logger.log(
        //     format!(
        //         "[BUILD({:?})-TXN]({}) - {} :: ({:?})",
        //         swap_config.swap_direction,
        //         mint_str,
        //         Utc::now(),
        //         start_time.elapsed()
        //     )
        //     .yellow()
        //     .to_string(),
        // );

        // Expire Condition
        // -------------------
        if swap_config.swap_direction == SwapDirection::Buy
            && start_time.elapsed() > Duration::from_millis(700)
        {
            return Err(anyhow!("RPC connection is too busy. Expire this txn."
                .red()
                .italic()
                .to_string()));
        }

        // Return- (instructions, token_price)
        // --------------------
        Ok((self.keypair.clone(), instructions, token_price))
    }
}

fn min_amount_with_slippage(input_amount: u64, slippage_bps: u64) -> u64 {
    input_amount
        .checked_mul(TEN_THOUSAND.checked_sub(slippage_bps).unwrap())
        .unwrap()
        .checked_div(TEN_THOUSAND)
        .unwrap()
}
fn max_amount_with_slippage(input_amount: u64, slippage_bps: u64) -> u64 {
    input_amount
        .checked_mul(slippage_bps.checked_add(TEN_THOUSAND).unwrap())
        .unwrap()
        .checked_div(TEN_THOUSAND)
        .unwrap()
}
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaydiumInfo {
    pub base: f64,
    pub quote: f64,
    pub price: f64,
}
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PumpInfo {
    pub mint: String,
    pub bonding_curve: String,
    pub associated_bonding_curve: String,
    pub raydium_pool: Option<String>,
    pub raydium_info: Option<RaydiumInfo>,
    pub complete: bool,
    pub virtual_sol_reserves: u64,
    pub virtual_token_reserves: u64,
    pub total_supply: u64,
}

#[derive(Debug, BorshSerialize, BorshDeserialize)]
pub struct BondingCurveAccount {
    pub discriminator: u64,
    pub virtual_token_reserves: u64,
    pub virtual_sol_reserves: u64,
    pub real_token_reserves: u64,
    pub real_sol_reserves: u64,
    pub token_total_supply: u64,
    pub complete: bool,
}

#[derive(Debug, BorshSerialize, BorshDeserialize)]
pub struct BondingCurveReserves {
    pub virtual_token_reserves: u64,
    pub virtual_sol_reserves: u64,
}

pub fn get_bonding_curve_account_by_calc(
    bonding_curve_info: BondingCurveInfo,
    mint: Pubkey,
) -> (Pubkey, Pubkey, BondingCurveReserves) {
    let bonding_curve = bonding_curve_info.bonding_curve;
    let associated_bonding_curve = get_associated_token_address(&bonding_curve, &mint);
    
    let bonding_curve_reserves = BondingCurveReserves 
        { 
            virtual_token_reserves: bonding_curve_info.new_virtual_token_reserve, 
            virtual_sol_reserves: bonding_curve_info.new_virtual_sol_reserve,
        };

    (
        bonding_curve,
        associated_bonding_curve,
        bonding_curve_reserves,
    )
}

pub async fn get_bonding_curve_account(
    rpc_client: Arc<anchor_client::solana_client::rpc_client::RpcClient>,
    mint: Pubkey,
    program_id: Pubkey,
) -> Result<(Pubkey, Pubkey, BondingCurveReserves)> {
    let bonding_curve = get_pda(&mint, &program_id)?;
    let associated_bonding_curve = get_associated_token_address(&bonding_curve, &mint);
    let start_time = Instant::now();
    // println!("Start: {:?}", start_time.elapsed());

    let max_retries = 30;
    let time_exceed = 300;
    let timeout = Duration::from_millis(time_exceed);
    let mut retry_count = 0;
    let bonding_curve_data = loop {
        match rpc_client.get_account_data(&bonding_curve) {
            Ok(data) => {
                // println!("Done: {:?}", start_time.elapsed());
                break data;
            }
            Err(err) => {
                retry_count += 1;
                if retry_count > max_retries {
                    return Err(anyhow!(
                        "Failed to get bonding curve account data after {} retries: {}",
                        max_retries,
                        err
                    ));
                }
                if start_time.elapsed() > timeout {
                    return Err(anyhow!(
                        "Failed to get bonding curve account data after {:?} timeout: {}",
                        timeout,
                        err
                    ));
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
                // println!("Retry {}: {:?}", retry_count, start_time.elapsed());
            }
        }
    };

    let bonding_curve_account =
        from_slice::<BondingCurveAccount>(&bonding_curve_data).map_err(|e| {
            anyhow!(
                "Failed to deserialize bonding curve account: {}",
                e.to_string()
            )
        })?;
    let bonding_curve_reserves = BondingCurveReserves 
        { 
            virtual_token_reserves: bonding_curve_account.virtual_token_reserves, 
            virtual_sol_reserves: bonding_curve_account.virtual_sol_reserves 
        };
    Ok((
        bonding_curve,
        associated_bonding_curve,
        bonding_curve_reserves,
    ))
}

pub fn get_pda(mint: &Pubkey, program_id: &Pubkey) -> Result<Pubkey> {
    let seeds = [b"bonding-curve".as_ref(), mint.as_ref()];
    let (bonding_curve, _bump) = Pubkey::find_program_address(&seeds, program_id);
    Ok(bonding_curve)
}

// https://frontend-api.pump.fun/coins/8zSLdDzM1XsqnfrHmHvA9ir6pvYDjs8UXz6B2Tydd6b2
// pub async fn get_pump_info(
//     rpc_client: Arc<solana_client::rpc_client::RpcClient>,
//     mint: str,
// ) -> Result<PumpInfo> {
//     let mint = Pubkey::from_str(&mint)?;
//     let program_id = Pubkey::from_str(PUMP_PROGRAM)?;
//     let (bonding_curve, associated_bonding_curve, bonding_curve_account) =
//         get_bonding_curve_account(rpc_client, &mint, &program_id).await?;

//     let pump_info = PumpInfo {
//         mint: mint.to_string(),
//         bonding_curve: bonding_curve.to_string(),
//         associated_bonding_curve: associated_bonding_curve.to_string(),
//         raydium_pool: None,
//         raydium_info: None,
//         complete: bonding_curve_account.complete,
//         virtual_sol_reserves: bonding_curve_account.virtual_sol_reserves,
//         virtual_token_reserves: bonding_curve_account.virtual_token_reserves,
//         total_supply: bonding_curve_account.token_total_supply,
//     };
//     Ok(pump_info)
// }
