//! Program state processor
#![allow(deprecated)]
use crate::{
    error::AmmError,
    instruction::{
        AdminCancelOrdersInstruction, AmmInstruction, ConfigArgs, DepositInstruction,
        InitializeInstruction2, MonitorStepInstruction, SetParamsInstruction, SimulateInstruction,
        SwapInstructionBaseIn, SwapInstructionBaseOut, WithdrawInstruction, WithdrawSrmInstruction,
    },
    invokers::Invokers,
    math::{
        Calculator, CheckedCeilDiv, InvariantPool, InvariantToken, RoundDirection, SwapDirection,
        U128, U256,
    },
    state::{
        AmmConfig, AmmInfo, AmmParams, AmmResetFlag, AmmState, AmmStatus, GetPoolData,
        GetSwapBaseInData, GetSwapBaseOutData, Loadable, RunCrankData, SimulateParams,
        TargetOrders, MAX_ORDER_LIMIT, TEN_THOUSAND,
    },
};

use serum_dex::{
    critbit::{LeafNode, Slab, SlabView},
    matching::{OrderType, Side},
    state::{Market, MarketState, OpenOrders, ToAlignedBytes},
};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    msg,
    program::{invoke, invoke_signed},
    // log::sol_log_compute_units,
    program_error::ProgramError,
    program_option::COption,
    program_pack::Pack,
    pubkey::Pubkey,
    rent::Rent,
    system_instruction,
    sysvar::{clock, Sysvar},
};

use super::log::*;
use arrayref::{array_ref, array_refs};
use arrform::{arrform, ArrForm};
use bytemuck::from_bytes_mut;
use std::{
    cell::RefMut, collections::VecDeque, convert::identity, convert::TryFrom, mem::size_of,
    num::NonZeroU64, ops::Deref,
};

pub mod srm_token {
    solana_program::declare_id!("SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt");
}

pub mod msrm_token {
    solana_program::declare_id!("MSRMcoVyrFxnSgo5uXwone5SKcGhT1KEJMFEkMEWf9L");
}

#[cfg(feature = "localnet")]
pub mod config_feature {
    pub mod amm_owner {
        solana_program::declare_id!("75KWb5XcqPTgacQyNw9P5QU2HL3xpezEVcgsFCiJgTT");
    }
    pub mod openbook_program {
        solana_program::declare_id!("kGeitTdTHT1WdpUScdm8yxUAirZwbnQtqrpzvAm1p98");
    }
    pub mod referrer_pc_wallet {
        solana_program::declare_id!("75KWb5XcqPTgacQyNw9P5QU2HL3xpezEVcgsFCiJgTT");
    }
    pub mod create_pool_fee_address {
        solana_program::declare_id!("75KWb5XcqPTgacQyNw9P5QU2HL3xpezEVcgsFCiJgTT");
    }
}
#[cfg(feature = "devnet")]
pub mod config_feature {
    pub mod amm_owner {
        solana_program::declare_id!("Adm29NctkKwJGaaiU8CXqdV6WDTwR81JbxV8zoxn745Y");
    }
    pub mod openbook_program {
        solana_program::declare_id!("EoTcMgcDRTJVZDMZWBoU6rhYHZfkNTVEAfz3uUJRcYGj");
    }
    pub mod referrer_pc_wallet {
        solana_program::declare_id!("4NpMfWThvJQsV9VLjUXXpn3tPv1zoQpib8wCBDc1EBzD");
    }
    pub mod create_pool_fee_address {
        solana_program::declare_id!("3XMrhbv989VxAMi3DErLV9eJht1pHppW5LbKxe9fkEFR");
    }
}
#[cfg(not(any(feature = "localnet", feature = "devnet")))]
pub mod config_feature {
    pub mod amm_owner {
        solana_program::declare_id!("GThUX1Atko4tqhN2NaiTazWSeFWMuiUvfFnyJyUghFMJ");
    }
    pub mod openbook_program {
        solana_program::declare_id!("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX");
    }
    pub mod referrer_pc_wallet {
        solana_program::declare_id!("FCxGKqGSVeV1d3WsmAXt45A5iQdCS6kKCeJy3EUBigMG");
    }
    pub mod create_pool_fee_address {
        solana_program::declare_id!("7YttLkHDoNj9wyDur5pM1ejNaAvT9X4eqaYcHQqtj2G5");
    }
}

/// Suffix for amm authority seed
pub const AUTHORITY_AMM: &'static [u8] = b"amm authority";
/// Suffix for amm associated seed
pub const AMM_ASSOCIATED_SEED: &'static [u8] = b"amm_associated_seed";
/// Suffix for target associated seed
pub const TARGET_ASSOCIATED_SEED: &'static [u8] = b"target_associated_seed";
/// Suffix for amm open order associated seed
pub const OPEN_ORDER_ASSOCIATED_SEED: &'static [u8] = b"open_order_associated_seed";
/// Suffix for coin vault associated seed
pub const COIN_VAULT_ASSOCIATED_SEED: &'static [u8] = b"coin_vault_associated_seed";
/// Suffix for pc vault associated seed
pub const PC_VAULT_ASSOCIATED_SEED: &'static [u8] = b"pc_vault_associated_seed";
/// Suffix for lp mint associated seed
pub const LP_MINT_ASSOCIATED_SEED: &'static [u8] = b"lp_mint_associated_seed";
/// Amm config seed
pub const AMM_CONFIG_SEED: &'static [u8] = b"amm_config_account_seed";

pub fn get_associated_address_and_bump_seed(
    info_id: &Pubkey,
    market_address: &Pubkey,
    associated_seed: &[u8],
    program_id: &Pubkey,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            &info_id.to_bytes(),
            &market_address.to_bytes(),
            &associated_seed,
        ],
        program_id,
    )
}

/// Program state handler.
pub struct Processor {}
impl Processor {
    #[inline]
    fn check_account_readonly(account_info: &AccountInfo) -> ProgramResult {
        if account_info.is_writable {
            return Err(AmmError::AccountNeedReadOnly.into());
        }
        return Ok(());
    }

    /// Unpacks a spl_token `Account`.
    #[inline]
    pub fn unpack_token_account(
        account_info: &AccountInfo,
        token_program_id: &Pubkey,
    ) -> Result<spl_token::state::Account, AmmError> {
        if account_info.owner != token_program_id {
            Err(AmmError::InvalidSplTokenProgram)
        } else {
            spl_token::state::Account::unpack(&account_info.data.borrow())
                .map_err(|_| AmmError::ExpectedAccount)
        }
    }

    /// Unpacks a spl_token `Mint`.
    #[inline]
    pub fn unpack_mint(
        account_info: &AccountInfo,
        token_program_id: &Pubkey,
    ) -> Result<spl_token::state::Mint, AmmError> {
        if account_info.owner != token_program_id {
            Err(AmmError::InvalidSplTokenProgram)
        } else {
            spl_token::state::Mint::unpack(&account_info.data.borrow())
                .map_err(|_| AmmError::ExpectedMint)
        }
    }

    fn load_orders<'a>(
        orders_account: &'a AccountInfo,
    ) -> Result<RefMut<'a, OpenOrders>, ProgramError> {
        let (_, data) = serum_dex::state::strip_header::<[u8; 0], u8>(orders_account, false)?;
        let open_orders: RefMut<'a, OpenOrders>;
        open_orders = RefMut::map(data, |data| from_bytes_mut(data));
        Ok(open_orders)
    }

    pub fn load_serum_market_order<'a>(
        market_acc: &AccountInfo<'a>,
        open_orders_acc: &AccountInfo<'a>,
        authority_acc: &AccountInfo<'a>,
        amm: &AmmInfo,
        // Allow for the market flag to be set to AccountFlag::Disabled
        allow_disabled: bool,
    ) -> Result<(Box<MarketState>, Box<OpenOrders>), ProgramError> {
        let market_state = Market::load(market_acc, &amm.market_program, allow_disabled)?;
        let open_orders = market_state.load_orders_mut(
            open_orders_acc,
            Some(authority_acc),
            &amm.market_program,
            None,
            None,
        )?;
        if identity(open_orders.market) != market_acc.key.to_aligned_bytes() {
            return Err(AmmError::InvalidMarket.into());
        }
        if identity(open_orders.owner) != authority_acc.key.to_aligned_bytes() {
            return Err(AmmError::InvalidOwner.into());
        }
        if *open_orders_acc.key != amm.open_orders {
            return Err(AmmError::InvalidOpenOrders.into());
        }
        return Ok((
            Box::new(*market_state.deref()),
            Box::new(*open_orders.deref()),
        ));
    }

    fn _get_dex_best_price(slab: &RefMut<serum_dex::critbit::Slab>, side: Side) -> Option<u64> {
        if slab.is_empty() {
            None
        } else {
            match side {
                Side::Bid => {
                    let best_bid_h = slab.find_max().unwrap();
                    let best_bid_px = slab
                        .get(best_bid_h)
                        .unwrap()
                        .as_leaf()
                        .unwrap()
                        .price()
                        .get();
                    Some(best_bid_px)
                }
                Side::Ask => {
                    let best_ask_h = slab.find_min().unwrap();
                    let best_ask_px = slab
                        .get(best_ask_h)
                        .unwrap()
                        .as_leaf()
                        .unwrap()
                        .price()
                        .get();
                    Some(best_ask_px)
                }
            }
        }
    }

    fn get_amm_orders(
        open_orders: &OpenOrders,
        bids: RefMut<Slab>,
        asks: RefMut<Slab>,
    ) -> Result<(Vec<LeafNode>, Vec<LeafNode>), ProgramError> {
        let orders_number = open_orders.free_slot_bits.count_zeros();
        let mut bids_orders: Vec<LeafNode> = Vec::new();
        let mut asks_orders: Vec<LeafNode> = Vec::new();
        if orders_number != 0 {
            for i in 0..128 {
                let slot_mask = 1u128 << i;
                if open_orders.free_slot_bits & slot_mask != 0 {
                    // means slot is free
                    continue;
                }
                if open_orders.is_bid_bits & slot_mask != 0 {
                    match bids.find_by_key(open_orders.orders[i]) {
                        None => continue,
                        Some(handle_bid) => {
                            let handle_bid_ref = bids.get(handle_bid).unwrap().as_leaf().unwrap();
                            bids_orders.push(*handle_bid_ref);
                        }
                    }
                } else {
                    match asks.find_by_key(open_orders.orders[i]) {
                        None => continue,
                        Some(handle_ask) => {
                            let handle_ask_ref = asks.get(handle_ask).unwrap().as_leaf().unwrap();
                            asks_orders.push(*handle_ask_ref);
                        }
                    }
                };
            }
        }
        bids_orders.sort_by(|a, b| b.price().get().cmp(&a.price().get()));
        asks_orders.sort_by(|a, b| a.price().get().cmp(&b.price().get()));
        Ok((bids_orders, asks_orders))
    }

    pub fn get_amm_best_price(
        market_state: &MarketState,
        open_orders: &OpenOrders,
        bids_account: &AccountInfo,
        asks_account: &AccountInfo,
    ) -> Result<(Option<u64>, Option<u64>), ProgramError> {
        let bids = market_state.load_bids_mut(&bids_account)?;
        let asks = market_state.load_asks_mut(&asks_account)?;
        let (bids_orders, asks_orders) = Self::get_amm_orders(open_orders, bids, asks)?;
        let mut bid_best_price = None;
        let mut ask_best_price = None;
        if bids_orders.len() != 0 {
            bid_best_price = Some(bids_orders.first().unwrap().price().get());
        }
        if asks_orders.len() != 0 {
            ask_best_price = Some(asks_orders.first().unwrap().price().get());
        }
        Ok((bid_best_price, ask_best_price))
    }

    pub fn get_amm_worst_price(
        market_state: &MarketState,
        open_orders: &OpenOrders,
        bids_account: &AccountInfo,
        asks_account: &AccountInfo,
    ) -> Result<(Option<u64>, Option<u64>), ProgramError> {
        let bids = market_state.load_bids_mut(&bids_account)?;
        let asks = market_state.load_asks_mut(&asks_account)?;
        let (bids_orders, asks_orders) = Self::get_amm_orders(open_orders, bids, asks)?;
        let mut bid_best_price = None;
        let mut ask_best_price = None;
        if bids_orders.len() != 0 {
            bid_best_price = Some(bids_orders.last().unwrap().price().get());
        }
        if asks_orders.len() != 0 {
            ask_best_price = Some(asks_orders.last().unwrap().price().get());
        }
        Ok((bid_best_price, ask_best_price))
    }

    /// The Detailed calculation of pnl
    /// 1. calc last_k witch dose not take pnl: last_k = calc_pnl_x * calc_pnl_y;
    /// 2. calc current price: current_price = current_x / current_y;
    /// 3. calc x after take pnl: x_after_take_pnl = sqrt(last_k * current_price);
    /// 4. calc y after take pnl: y_after_take_pnl = x_after_take_pnl / current_price;
    ///                           y_after_take_pnl = x_after_take_pnl * current_y / current_x;
    /// 5. calc pnl_x & pnl_y:  pnl_x = current_x - x_after_take_pnl;
    ///                         pnl_y = current_y - y_after_take_pnl;
    pub fn calc_take_pnl(
        target: &TargetOrders,
        amm: &mut AmmInfo,
        total_pc_without_take_pnl: &mut u64,
        total_coin_without_take_pnl: &mut u64,
        x1: U256,
        y1: U256,
    ) -> Result<(u128, u128), ProgramError> {
        // calc pnl
        let mut delta_x: u128;
        let mut delta_y: u128;
        let calc_pc_amount = Calculator::restore_decimal(
            target.calc_pnl_x.into(),
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let calc_coin_amount = Calculator::restore_decimal(
            target.calc_pnl_y.into(),
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        let pool_pc_amount = U128::from(*total_pc_without_take_pnl);
        let pool_coin_amount = U128::from(*total_coin_without_take_pnl);
        if pool_pc_amount.checked_mul(pool_coin_amount).unwrap()
            >= (calc_pc_amount).checked_mul(calc_coin_amount).unwrap()
        {
            // last k is
            // let last_k: u128 = (target.calc_pnl_x as u128).checked_mul(target.calc_pnl_y as u128).unwrap();
            // current k is
            // let current_k: u128 = (x1 as u128).checked_mul(y1 as u128).unwrap();
            // current p is
            // let current_p: u128 = (x1 as u128).checked_div(y1 as u128).unwrap();
            let x2_power = Calculator::calc_x_power(
                target.calc_pnl_x.into(),
                target.calc_pnl_y.into(),
                x1,
                y1,
            );
            // let x2 = Calculator::sqrt(x2_power).unwrap();
            let x2 = x2_power.integer_sqrt();
            // msg!(arrform!(LOG_SIZE, "calc_take_pnl x2_power:{}, x2:{}", x2_power, x2).as_str());
            let y2 = x2.checked_mul(y1).unwrap().checked_div(x1).unwrap();
            // msg!(arrform!(LOG_SIZE, "calc_take_pnl y2:{}", y2).as_str());

            // transfer to token_coin_pnl and token_pc_pnl
            // (x1 -x2) * pnl / sys_decimal_value
            let diff_x = U128::from(x1.checked_sub(x2).unwrap().as_u128());
            let diff_y = U128::from(y1.checked_sub(y2).unwrap().as_u128());
            delta_x = diff_x
                .checked_mul(amm.fees.pnl_numerator.into())
                .unwrap()
                .checked_div(amm.fees.pnl_denominator.into())
                .unwrap()
                .as_u128();
            delta_y = diff_y
                .checked_mul(amm.fees.pnl_numerator.into())
                .unwrap()
                .checked_div(amm.fees.pnl_denominator.into())
                .unwrap()
                .as_u128();

            let diff_pc_pnl_amount =
                Calculator::restore_decimal(diff_x, amm.pc_decimals, amm.sys_decimal_value);
            let diff_coin_pnl_amount =
                Calculator::restore_decimal(diff_y, amm.coin_decimals, amm.sys_decimal_value);
            let pc_pnl_amount = diff_pc_pnl_amount
                .checked_mul(amm.fees.pnl_numerator.into())
                .unwrap()
                .checked_div(amm.fees.pnl_denominator.into())
                .unwrap()
                .as_u64();
            let coin_pnl_amount = diff_coin_pnl_amount
                .checked_mul(amm.fees.pnl_numerator.into())
                .unwrap()
                .checked_div(amm.fees.pnl_denominator.into())
                .unwrap()
                .as_u64();
            if pc_pnl_amount != 0 && coin_pnl_amount != 0 {
                // step2: save total_pnl_pc & total_pnl_coin
                amm.state_data.total_pnl_pc = amm
                    .state_data
                    .total_pnl_pc
                    .checked_add(diff_pc_pnl_amount.as_u64())
                    .unwrap();
                amm.state_data.total_pnl_coin = amm
                    .state_data
                    .total_pnl_coin
                    .checked_add(diff_coin_pnl_amount.as_u64())
                    .unwrap();
                amm.state_data.need_take_pnl_pc = amm
                    .state_data
                    .need_take_pnl_pc
                    .checked_add(pc_pnl_amount)
                    .unwrap();
                amm.state_data.need_take_pnl_coin = amm
                    .state_data
                    .need_take_pnl_coin
                    .checked_add(coin_pnl_amount)
                    .unwrap();

                // step3: update total_coin and total_pc without pnl
                *total_pc_without_take_pnl = (*total_pc_without_take_pnl)
                    .checked_sub(pc_pnl_amount)
                    .unwrap();
                *total_coin_without_take_pnl = (*total_coin_without_take_pnl)
                    .checked_sub(coin_pnl_amount)
                    .unwrap();
            } else {
                delta_x = 0;
                delta_y = 0;
            }
        } else {
            msg!(arrform!(
                LOG_SIZE,
                "calc_take_pnl error x:{}, y:{}, calc_pnl_x:{}, calc_pnl_y:{}",
                x1,
                y1,
                target.calc_pnl_x,
                target.calc_pnl_y
            )
            .as_str());
            return Err(AmmError::CalcPnlError.into());
        }

        Ok((delta_x, delta_y))
    }

    /// Calculates the authority id by generating a program address.
    pub fn authority_id(
        program_id: &Pubkey,
        amm_seed: &[u8],
        nonce: u8,
    ) -> Result<Pubkey, AmmError> {
        Pubkey::create_program_address(&[amm_seed, &[nonce]], program_id)
            .map_err(|_| AmmError::InvalidProgramAddress.into())
    }

    #[allow(clippy::too_many_arguments)]
    fn check_accounts(
        program_id: &Pubkey,
        amm: &AmmInfo,
        amm_info: &AccountInfo,
        token_program_info: &AccountInfo,
        clock_info: &AccountInfo,
        market_program_info: &AccountInfo,
        amm_authority_info: &AccountInfo,
        market_info: &AccountInfo,
        amm_open_orders_info: &AccountInfo,
        amm_coin_vault_info: &AccountInfo,
        amm_pc_vault_info: &AccountInfo,
        amm_target_orders_info: &AccountInfo,
        srm_token_info: Option<&AccountInfo>,
        referrer_pc_info: Option<&AccountInfo>,
    ) -> ProgramResult {
        if amm.status == AmmStatus::Uninitialized.into_u64() {
            return Err(AmmError::InvalidStatus.into());
        }
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        check_assert_eq!(
            *amm_info.owner,
            *program_id,
            "amm_owner",
            AmmError::InvalidOwner
        );
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let token_program_id = token_program_info.key;
        check_assert_eq!(
            *clock_info.key,
            clock::id(),
            "clock",
            AmmError::InvalidProgramAddress
        );
        check_assert_eq!(
            *market_program_info.key,
            amm.market_program,
            "market_program",
            AmmError::InvalidMarketProgram
        );
        check_assert_eq!(
            *market_info.key,
            amm.market,
            "market_info",
            AmmError::InvalidMarket
        );
        check_assert_eq!(
            *amm_open_orders_info.key,
            amm.open_orders,
            "open_order",
            AmmError::InvalidOpenOrders
        );
        check_assert_eq!(
            *amm_coin_vault_info.key,
            amm.coin_vault,
            "coin_vault",
            AmmError::InvalidCoinVault
        );
        check_assert_eq!(
            *amm_pc_vault_info.key,
            amm.pc_vault,
            "pc_vault",
            AmmError::InvalidPCVault
        );
        check_assert_eq!(
            *amm_target_orders_info.key,
            amm.target_orders,
            "target_orders",
            AmmError::InvalidTargetOrders
        );
        if let Some(srm_token_account) = srm_token_info {
            let srm_token = Self::unpack_token_account(&srm_token_account, token_program_id)?;
            check_assert_eq!(
                srm_token.owner,
                *amm_authority_info.key,
                "srm_token_owner",
                AmmError::InvalidOwner
            );
            if srm_token.mint != srm_token::id() && srm_token.mint != msrm_token::id() {
                return Err(AmmError::InvalidSrmMint.into());
            }
        }
        if let Some(referrer_pc_account) = referrer_pc_info {
            let referrer_pc_token =
                Self::unpack_token_account(&referrer_pc_account, token_program_id)?;
            check_assert_eq!(
                referrer_pc_token.owner,
                config_feature::referrer_pc_wallet::id(),
                "referrer_pc_owner",
                AmmError::InvalidOwner
            );
            check_assert_eq!(
                referrer_pc_token.mint,
                amm.pc_vault_mint,
                "referrer_pc",
                AmmError::InvalidReferPCMint
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_amm_associated_spl_token<'a, 'b: 'a>(
        program_id: &Pubkey,
        spl_token_program_id: &Pubkey,
        market_account: &'a AccountInfo<'b>,
        associated_token_account: &'a AccountInfo<'b>,
        token_mint_account: &'a AccountInfo<'b>,
        user_wallet_account: &'a AccountInfo<'b>,
        system_program_account: &'a AccountInfo<'b>,
        rent_sysvar_account: &'a AccountInfo<'b>,
        spl_token_program_account: &'a AccountInfo<'b>,
        associated_owner_account: &'a AccountInfo<'b>,
        associated_seed: &[u8],
    ) -> ProgramResult {
        let (associated_token_address, bump_seed) = get_associated_address_and_bump_seed(
            program_id,
            &market_account.key,
            associated_seed,
            program_id,
        );
        if associated_token_address != *associated_token_account.key {
            msg!("Error: Associated token address does not match seed derivation");
            return Err(AmmError::ExpectedAccount.into());
        }
        if associated_token_account.owner == system_program_account.key {
            let associated_account_signer_seeds: &[&[_]] = &[
                &program_id.to_bytes(),
                &market_account.key.to_bytes(),
                associated_seed,
                &[bump_seed],
            ];
            let rent = &Rent::from_account_info(rent_sysvar_account)?;
            let required_lamports = rent
                .minimum_balance(spl_token::state::Account::LEN)
                .max(1)
                .saturating_sub(associated_token_account.lamports());
            if required_lamports > 0 {
                invoke(
                    &system_instruction::transfer(
                        user_wallet_account.key,
                        associated_token_account.key,
                        required_lamports,
                    ),
                    &[
                        user_wallet_account.clone(),
                        associated_token_account.clone(),
                        system_program_account.clone(),
                    ],
                )?;
            }
            invoke_signed(
                &system_instruction::allocate(
                    associated_token_account.key,
                    spl_token::state::Account::LEN as u64,
                ),
                &[
                    associated_token_account.clone(),
                    system_program_account.clone(),
                ],
                &[&associated_account_signer_seeds],
            )?;
            invoke_signed(
                &system_instruction::assign(associated_token_account.key, spl_token_program_id),
                &[
                    associated_token_account.clone(),
                    system_program_account.clone(),
                ],
                &[&associated_account_signer_seeds],
            )?;

            invoke(
                &spl_token::instruction::initialize_account(
                    spl_token_program_id,
                    associated_token_account.key,
                    token_mint_account.key,
                    associated_owner_account.key,
                )?,
                &[
                    associated_token_account.clone(),
                    token_mint_account.clone(),
                    associated_owner_account.clone(),
                    rent_sysvar_account.clone(),
                    spl_token_program_account.clone(),
                ],
            )?;
        } else {
            associated_token_address.log();
            return Err(AmmError::RepeatCreateAmm.into());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_amm_associated_spl_mint<'a, 'b: 'a>(
        program_id: &Pubkey,
        spl_token_program_id: &Pubkey,
        market_account: &'a AccountInfo<'b>,
        associated_token_account: &'a AccountInfo<'b>,
        user_wallet_account: &'a AccountInfo<'b>,
        system_program_account: &'a AccountInfo<'b>,
        rent_sysvar_account: &'a AccountInfo<'b>,
        spl_token_program_account: &'a AccountInfo<'b>,
        associated_owner_account: &'a AccountInfo<'b>,
        associated_seed: &[u8],
        mint_decimals: u8,
    ) -> ProgramResult {
        let (associated_token_address, bump_seed) = get_associated_address_and_bump_seed(
            program_id,
            &market_account.key,
            associated_seed,
            program_id,
        );
        if associated_token_address != *associated_token_account.key {
            msg!("Error: Associated mint address does not match seed derivation");
            return Err(AmmError::ExpectedMint.into());
        }
        if associated_token_account.owner == system_program_account.key {
            let associated_account_signer_seeds: &[&[_]] = &[
                &program_id.to_bytes(),
                &market_account.key.to_bytes(),
                associated_seed,
                &[bump_seed],
            ];
            let rent = &Rent::from_account_info(rent_sysvar_account)?;
            let required_lamports = rent
                .minimum_balance(spl_token::state::Mint::LEN)
                .max(1)
                .saturating_sub(associated_token_account.lamports());
            if required_lamports > 0 {
                invoke(
                    &system_instruction::transfer(
                        user_wallet_account.key,
                        associated_token_account.key,
                        required_lamports,
                    ),
                    &[
                        user_wallet_account.clone(),
                        associated_token_account.clone(),
                        system_program_account.clone(),
                    ],
                )?;
            }
            invoke_signed(
                &system_instruction::allocate(
                    associated_token_account.key,
                    spl_token::state::Mint::LEN as u64,
                ),
                &[
                    associated_token_account.clone(),
                    system_program_account.clone(),
                ],
                &[&associated_account_signer_seeds],
            )?;
            invoke_signed(
                &system_instruction::assign(associated_token_account.key, spl_token_program_id),
                &[
                    associated_token_account.clone(),
                    system_program_account.clone(),
                ],
                &[&associated_account_signer_seeds],
            )?;

            invoke(
                &spl_token::instruction::initialize_mint(
                    spl_token_program_id,
                    associated_token_account.key,
                    associated_owner_account.key,
                    None,
                    mint_decimals,
                )?,
                &[
                    associated_token_account.clone(),
                    associated_owner_account.clone(),
                    rent_sysvar_account.clone(),
                    spl_token_program_account.clone(),
                ],
            )?;
        } else {
            associated_token_address.log();
            return Err(AmmError::RepeatCreateAmm.into());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_amm_associated_account<'a, 'b: 'a>(
        program_id: &Pubkey,
        assign_to: &Pubkey,
        market_account: &'a AccountInfo<'b>,
        associated_token_account: &'a AccountInfo<'b>,
        user_wallet_account: &'a AccountInfo<'b>,
        system_program_account: &'a AccountInfo<'b>,
        rent_sysvar_account: &'a AccountInfo<'b>,
        associated_seed: &[u8],
        data_size: usize,
    ) -> ProgramResult {
        let (associated_token_address, bump_seed) = get_associated_address_and_bump_seed(
            &program_id,
            &market_account.key,
            associated_seed,
            program_id,
        );
        if associated_token_address != *associated_token_account.key {
            msg!("Error: Associated token address does not match seed derivation");
            return Err(AmmError::ExpectedAccount.into());
        }
        if associated_token_account.owner == system_program_account.key {
            let associated_account_signer_seeds: &[&[_]] = &[
                &program_id.to_bytes(),
                &market_account.key.to_bytes(),
                associated_seed,
                &[bump_seed],
            ];
            let rent = &Rent::from_account_info(rent_sysvar_account)?;
            let required_lamports = rent
                .minimum_balance(data_size)
                .max(1)
                .saturating_sub(associated_token_account.lamports());
            if required_lamports > 0 {
                invoke(
                    &system_instruction::transfer(
                        user_wallet_account.key,
                        associated_token_account.key,
                        required_lamports,
                    ),
                    &[
                        user_wallet_account.clone(),
                        associated_token_account.clone(),
                        system_program_account.clone(),
                    ],
                )?;
            }
            invoke_signed(
                &system_instruction::allocate(associated_token_account.key, data_size as u64),
                &[
                    associated_token_account.clone(),
                    system_program_account.clone(),
                ],
                &[&associated_account_signer_seeds],
            )?;
            invoke_signed(
                &system_instruction::assign(associated_token_account.key, assign_to),
                &[
                    associated_token_account.clone(),
                    system_program_account.clone(),
                ],
                &[&associated_account_signer_seeds],
            )?;
        } else {
            associated_token_address.log();
            return Err(AmmError::RepeatCreateAmm.into());
        }
        Ok(())
    }

    /// Processes an [Initialize](enum.Instruction.html).
    pub fn process_initialize2(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        init: InitializeInstruction2,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;
        let ata_token_program_info = next_account_info(account_info_iter)?;
        let system_program_info = next_account_info(account_info_iter)?;
        let rent_sysvar_info = next_account_info(account_info_iter)?;
        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_lp_mint_info = next_account_info(account_info_iter)?;
        let amm_coin_mint_info = next_account_info(account_info_iter)?;
        let amm_pc_mint_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_config_info = next_account_info(account_info_iter)?;
        let create_fee_destination_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;

        let user_wallet_info = next_account_info(account_info_iter)?;
        let user_token_coin_info = next_account_info(account_info_iter)?;
        let user_token_pc_info = next_account_info(account_info_iter)?;
        let user_token_lp_info = next_account_info(account_info_iter)?;

        let (pda, _) = Pubkey::find_program_address(&[&AMM_CONFIG_SEED], program_id);
        if pda != *amm_config_info.key || amm_config_info.owner != program_id {
            return Err(AmmError::InvalidConfigAccount.into());
        }

        msg!(arrform!(LOG_SIZE, "initialize2: {:?}", init).as_str());
        if !user_wallet_info.is_signer {
            return Err(AmmError::InvalidSignAccount.into());
        }
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        check_assert_eq!(
            *ata_token_program_info.key,
            spl_associated_token_account::id(),
            "spl_associated_token_account",
            AmmError::InvalidSplTokenProgram
        );
        check_assert_eq!(
            *market_program_info.key,
            config_feature::openbook_program::id(),
            "market_program",
            AmmError::InvalidMarketProgram
        );
        check_assert_eq!(
            *system_program_info.key,
            solana_program::system_program::id(),
            "sys_program",
            AmmError::InvalidSysProgramAddress
        );
        let (expect_amm_authority, expect_nonce) =
            Pubkey::find_program_address(&[&AUTHORITY_AMM], program_id);
        if *amm_authority_info.key != expect_amm_authority || init.nonce != expect_nonce {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        if *create_fee_destination_info.key != config_feature::create_pool_fee_address::id() {
            return Err(AmmError::InvalidFee.into());
        }
        let amm_config = AmmConfig::load_checked(&amm_config_info, program_id)?;
        // Charge the fee to create a pool
        if amm_config.create_pool_fee != 0 {
            invoke(
                &system_instruction::transfer(
                    user_wallet_info.key,
                    create_fee_destination_info.key,
                    amm_config.create_pool_fee,
                ),
                &[
                    user_wallet_info.clone(),
                    create_fee_destination_info.clone(),
                    system_program_info.clone(),
                ],
            )?;
            invoke(
                &spl_token::instruction::sync_native(
                    token_program_info.key,
                    create_fee_destination_info.key,
                )?,
                &[
                    token_program_info.clone(),
                    create_fee_destination_info.clone(),
                ],
            )?;
        }

        // unpack and check coin_mint
        let coin_mint = Self::unpack_mint(&amm_coin_mint_info, spl_token_program_id)?;
        // unpack and check pc_mint
        let pc_mint = Self::unpack_mint(&amm_pc_mint_info, spl_token_program_id)?;

        // create target_order account
        Self::generate_amm_associated_account(
            program_id,
            program_id,
            market_info,
            amm_target_orders_info,
            user_wallet_info,
            system_program_info,
            rent_sysvar_info,
            TARGET_ASSOCIATED_SEED,
            size_of::<TargetOrders>(),
        )?;

        // create lp mint account
        let lp_decimals = coin_mint.decimals;
        Self::generate_amm_associated_spl_mint(
            program_id,
            spl_token_program_id,
            market_info,
            amm_lp_mint_info,
            user_wallet_info,
            system_program_info,
            rent_sysvar_info,
            token_program_info,
            amm_authority_info,
            LP_MINT_ASSOCIATED_SEED,
            lp_decimals,
        )?;
        // create coin vault account
        Self::generate_amm_associated_spl_token(
            program_id,
            spl_token_program_id,
            market_info,
            amm_coin_vault_info,
            amm_coin_mint_info,
            user_wallet_info,
            system_program_info,
            rent_sysvar_info,
            token_program_info,
            amm_authority_info,
            COIN_VAULT_ASSOCIATED_SEED,
        )?;
        // create pc vault account
        Self::generate_amm_associated_spl_token(
            program_id,
            spl_token_program_id,
            market_info,
            amm_pc_vault_info,
            amm_pc_mint_info,
            user_wallet_info,
            system_program_info,
            rent_sysvar_info,
            token_program_info,
            amm_authority_info,
            PC_VAULT_ASSOCIATED_SEED,
        )?;
        // create amm account
        Self::generate_amm_associated_account(
            program_id,
            program_id,
            market_info,
            amm_info,
            user_wallet_info,
            system_program_info,
            rent_sysvar_info,
            AMM_ASSOCIATED_SEED,
            size_of::<AmmInfo>(),
        )?;

        // create amm open order account
        Self::generate_amm_associated_account(
            program_id,
            market_program_info.key,
            market_info,
            amm_open_orders_info,
            user_wallet_info,
            system_program_info,
            rent_sysvar_info,
            OPEN_ORDER_ASSOCIATED_SEED,
            size_of::<serum_dex::state::OpenOrders>() + 12,
        )?;
        // init open orders account
        Invokers::invoke_dex_init_open_orders(
            market_program_info.clone(),
            amm_open_orders_info.clone(),
            amm_authority_info.clone(),
            market_info.clone(),
            rent_sysvar_info.clone(),
            AUTHORITY_AMM,
            init.nonce as u8,
        )?;

        // create user ata lp token
        Invokers::create_ata_spl_token(
            user_token_lp_info.clone(),
            user_wallet_info.clone(),
            user_wallet_info.clone(),
            amm_lp_mint_info.clone(),
            token_program_info.clone(),
            ata_token_program_info.clone(),
            system_program_info.clone(),
        )?;

        // transfer user tokens to vault
        Invokers::token_transfer(
            token_program_info.clone(),
            user_token_coin_info.clone(),
            amm_coin_vault_info.clone(),
            user_wallet_info.clone(),
            init.init_coin_amount,
        )?;
        Invokers::token_transfer(
            token_program_info.clone(),
            user_token_pc_info.clone(),
            amm_pc_vault_info.clone(),
            user_wallet_info.clone(),
            init.init_pc_amount,
        )?;

        // load AmmInfo
        let mut amm = AmmInfo::load_mut(&amm_info)?;
        if amm.status != AmmStatus::Uninitialized.into_u64() {
            return Err(AmmError::AlreadyInUse.into());
        }

        // unpack and check token_coin
        let amm_coin_vault =
            Self::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
        check_assert_eq!(
            amm_coin_vault.owner,
            *amm_authority_info.key,
            "coin_vault_owner",
            AmmError::InvalidOwner
        );
        if amm_coin_vault.amount == 0 {
            return Err(AmmError::InvalidSupply.into());
        }
        if amm_coin_vault.delegate.is_some() {
            return Err(AmmError::InvalidDelegate.into());
        }
        if amm_coin_vault.close_authority.is_some() {
            return Err(AmmError::InvalidCloseAuthority.into());
        }
        check_assert_eq!(
            *amm_coin_mint_info.key,
            amm_coin_vault.mint,
            "coin_mint",
            AmmError::InvalidCoinMint
        );
        // unpack and check token_pc
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
        check_assert_eq!(
            amm_pc_vault.owner,
            *amm_authority_info.key,
            "pc_vault_owner",
            AmmError::InvalidOwner
        );
        if amm_pc_vault.amount == 0 {
            return Err(AmmError::InvalidSupply.into());
        }
        if amm_pc_vault.delegate.is_some() {
            return Err(AmmError::InvalidDelegate.into());
        }
        if amm_pc_vault.close_authority.is_some() {
            return Err(AmmError::InvalidCloseAuthority.into());
        }
        check_assert_eq!(
            *amm_pc_mint_info.key,
            amm_pc_vault.mint,
            "pc_mint",
            AmmError::InvalidPCMint
        );

        // load and check market
        let market_state =
            Market::load(market_info, &config_feature::openbook_program::id(), false)?;
        if identity(market_state.coin_mint) != amm_coin_vault.mint.to_aligned_bytes()
            || identity(market_state.coin_mint) != (*amm_coin_mint_info.key).to_aligned_bytes()
        {
            return Err(AmmError::InvalidCoinMint.into());
        }
        if identity(market_state.pc_mint) != amm_pc_vault.mint.to_aligned_bytes()
            || identity(market_state.pc_mint) != (*amm_pc_mint_info.key).to_aligned_bytes()
        {
            return Err(AmmError::InvalidPCMint.into());
        }
        if market_state.pc_lot_size == 0 || market_state.coin_lot_size == 0 {
            msg!(
                "pc_lot_size:{}, coin_lot_size:{}",
                identity(market_state.pc_lot_size),
                identity(market_state.coin_lot_size)
            );
            return Err(AmmError::InvalidMarket.into());
        }

        let lp_mint = Self::unpack_mint(&amm_lp_mint_info, spl_token_program_id)?;
        if lp_mint.supply != 0 {
            return Err(AmmError::InvalidSupply.into());
        }
        if COption::Some(*amm_authority_info.key) != lp_mint.mint_authority {
            return Err(AmmError::InvalidOwner.into());
        }
        if lp_mint.freeze_authority.is_some() {
            return Err(AmmError::InvalidFreezeAuthority.into());
        }

        let liquidity = Calculator::to_u64(
            U128::from(amm_pc_vault.amount)
                .checked_mul(amm_coin_vault.amount.into())
                .unwrap()
                .integer_sqrt()
                .as_u128(),
        )?;
        let user_lp_amount = liquidity
            .checked_sub((10u64).checked_pow(lp_mint.decimals.into()).unwrap())
            .ok_or(AmmError::InitLpAmountTooLess)?;

        // liquidity is measured in terms of token_a's value since both sides of
        // the pool are equal
        Invokers::token_mint_to(
            token_program_info.clone(),
            amm_lp_mint_info.clone(),
            user_token_lp_info.clone(),
            amm_authority_info.clone(),
            AUTHORITY_AMM,
            init.nonce,
            user_lp_amount,
        )?;

        amm.initialize(
            init.nonce,
            init.open_time,
            coin_mint.decimals,
            pc_mint.decimals,
            market_state.coin_lot_size,
            market_state.pc_lot_size,
        )?;
        encode_ray_log(InitLog {
            log_type: LogType::Init.into_u8(),
            time: init.open_time,
            pc_decimals: amm.pc_decimals as u8,
            coin_decimals: amm.coin_decimals as u8,
            pc_lot_size: market_state.pc_lot_size,
            coin_lot_size: market_state.coin_lot_size,
            pc_amount: amm_pc_vault.amount,
            coin_amount: amm_coin_vault.amount,
            market: *market_info.key,
        });
        let x = Calculator::normalize_decimal_v2(
            amm_pc_vault.amount,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y = Calculator::normalize_decimal_v2(
            amm_coin_vault.amount,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        // check and init target orders account
        if amm_target_orders_info.owner != program_id {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        let mut target_order = TargetOrders::load_mut(amm_target_orders_info)?;
        target_order.check_init(x.as_u128(), y.as_u128(), amm_info.key)?;

        amm.coin_vault = *amm_coin_vault_info.key;
        amm.pc_vault = *amm_pc_vault_info.key;
        amm.coin_vault_mint = *amm_coin_mint_info.key;
        amm.pc_vault_mint = *amm_pc_mint_info.key;
        amm.lp_mint = *amm_lp_mint_info.key;
        amm.open_orders = *amm_open_orders_info.key;
        amm.market = *market_info.key;
        amm.market_program = *market_program_info.key;
        amm.target_orders = *amm_target_orders_info.key;
        amm.amm_owner = config_feature::amm_owner::ID;
        amm.lp_amount = liquidity;
        amm.status = if init.open_time > (Clock::get()?.unix_timestamp as u64) {
            AmmStatus::WaitingTrade.into_u64()
        } else {
            AmmStatus::SwapOnly.into_u64()
        };
        amm.reset_flag = AmmResetFlag::ResetYes.into_u64();

        Ok(())
    }

    /// Processes an [Deposit](enum.Instruction.html).
    pub fn process_deposit(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        deposit: DepositInstruction,
    ) -> ProgramResult {
        const ACCOUNT_LEN: usize = 14;
        let input_account_len = accounts.len();
        if input_account_len != ACCOUNT_LEN && input_account_len != ACCOUNT_LEN + 1 {
            return Err(AmmError::WrongAccountsNumber.into());
        }
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_lp_mint_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;

        let market_info = next_account_info(account_info_iter)?;

        let user_source_coin_info = next_account_info(account_info_iter)?;
        let user_source_pc_info = next_account_info(account_info_iter)?;
        let user_dest_lp_info = next_account_info(account_info_iter)?;
        let source_owner_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;
        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        if deposit.max_coin_amount == 0 || deposit.max_pc_amount == 0 {
            encode_ray_log(DepositLog {
                log_type: LogType::Deposit.into_u8(),
                max_coin: deposit.max_coin_amount,
                max_pc: deposit.max_pc_amount,
                base: deposit.base_side,
                pool_coin: 0,
                pool_pc: 0,
                pool_lp: 0,
                calc_pnl_x: 0,
                calc_pnl_y: 0,
                deduct_coin: 0,
                deduct_pc: 0,
                mint_lp: 0,
            });
            return Err(AmmError::InvalidInput.into());
        }
        if !source_owner_info.is_signer {
            return Err(AmmError::InvalidSignAccount.into());
        }

        if !AmmStatus::from_u64(amm.status).deposit_permission() {
            return Err(AmmError::InvalidStatus.into());
        }
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        let enable_orderbook;
        if AmmStatus::from_u64(amm.status).orderbook_permission() {
            enable_orderbook = true;
        } else {
            enable_orderbook = false;
        }
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        // token_coin must be amm.coin_vault or token_source_coin must not be amm.coin_vault
        if *amm_coin_vault_info.key != amm.coin_vault
            || *user_source_coin_info.key == amm.coin_vault
        {
            return Err(AmmError::InvalidCoinVault.into());
        }
        // token_pc must be amm.pc_vault or token_source_pc must not be amm.pc_vault
        if *amm_pc_vault_info.key != amm.pc_vault || *user_source_pc_info.key == amm.pc_vault {
            return Err(AmmError::InvalidPCVault.into());
        }
        check_assert_eq!(
            *amm_lp_mint_info.key,
            amm.lp_mint,
            "lp_mint",
            AmmError::InvalidPoolMint
        );
        check_assert_eq!(
            *amm_target_orders_info.key,
            amm.target_orders,
            "target_orders",
            AmmError::InvalidTargetOrders
        );
        let amm_coin_vault =
            Self::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
        let user_source_coin =
            Self::unpack_token_account(&user_source_coin_info, spl_token_program_id)?;
        let user_source_pc =
            Self::unpack_token_account(&user_source_pc_info, spl_token_program_id)?;
        let mut target_orders =
            TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
        // calc the remaining total_pc & total_coin
        let (mut total_pc_without_take_pnl, mut total_coin_without_take_pnl) = if enable_orderbook {
            check_assert_eq!(
                *market_info.key,
                amm.market,
                "market",
                AmmError::InvalidMarket
            );
            check_assert_eq!(
                *amm_open_orders_info.key,
                amm.open_orders,
                "open_orders",
                AmmError::InvalidOpenOrders
            );
            let (market_state, open_orders) = Self::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;
            if identity(market_state.coin_mint) != amm_coin_vault.mint.to_aligned_bytes()
                || identity(market_state.coin_mint) != user_source_coin.mint.to_aligned_bytes()
            {
                return Err(AmmError::InvalidCoinMint.into());
            }
            if identity(market_state.pc_mint) != amm_pc_vault.mint.to_aligned_bytes()
                || identity(market_state.pc_mint) != user_source_pc.mint.to_aligned_bytes()
            {
                return Err(AmmError::InvalidPCMint.into());
            }
            Calculator::calc_total_without_take_pnl(
                amm_pc_vault.amount,
                amm_coin_vault.amount,
                &open_orders,
                &amm,
                &market_state,
                &market_event_queue_info,
                &amm_open_orders_info,
            )?
        } else {
            Calculator::calc_total_without_take_pnl_no_orderbook(
                amm_pc_vault.amount,
                amm_coin_vault.amount,
                &amm,
            )?
        };

        let x1 = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y1 = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        // calc and update pnl
        let (delta_x, delta_y) = Self::calc_take_pnl(
            &target_orders,
            &mut amm,
            &mut total_pc_without_take_pnl,
            &mut total_coin_without_take_pnl,
            x1.as_u128().into(),
            y1.as_u128().into(),
        )?;
        let invariant = InvariantToken {
            token_coin: total_coin_without_take_pnl,
            token_pc: total_pc_without_take_pnl,
        };

        // let lp_mint  = Self::unpack_mint(&lp_mint_info, spl_token_program_id)?;
        if amm.lp_amount == 0 {
            encode_ray_log(DepositLog {
                log_type: LogType::Deposit.into_u8(),
                max_coin: deposit.max_coin_amount,
                max_pc: deposit.max_pc_amount,
                base: deposit.base_side,
                pool_coin: total_coin_without_take_pnl,
                pool_pc: total_pc_without_take_pnl,
                pool_lp: amm.lp_amount,
                calc_pnl_x: target_orders.calc_pnl_x,
                calc_pnl_y: target_orders.calc_pnl_y,
                deduct_coin: 0,
                deduct_pc: 0,
                mint_lp: 0,
            });
            return Err(AmmError::NotAllowZeroLP.into());
        }
        let deduct_pc_amount;
        let deduct_coin_amount;
        let mint_lp_amount;
        if deposit.base_side == 0 {
            // base coin
            deduct_pc_amount = invariant
                .exchange_coin_to_pc(deposit.max_coin_amount, RoundDirection::Ceiling)
                .ok_or(AmmError::CalculationExRateFailure)?;
            deduct_coin_amount = deposit.max_coin_amount;
            if deduct_pc_amount > deposit.max_pc_amount {
                encode_ray_log(DepositLog {
                    log_type: LogType::Deposit.into_u8(),
                    max_coin: deposit.max_coin_amount,
                    max_pc: deposit.max_pc_amount,
                    base: deposit.base_side,
                    pool_coin: total_coin_without_take_pnl,
                    pool_pc: total_pc_without_take_pnl,
                    pool_lp: amm.lp_amount,
                    calc_pnl_x: target_orders.calc_pnl_x,
                    calc_pnl_y: target_orders.calc_pnl_y,
                    deduct_coin: deduct_coin_amount,
                    deduct_pc: deduct_pc_amount,
                    mint_lp: 0,
                });
                return Err(AmmError::ExceededSlippage.into());
            }
            // coin_amount/ (total_coin_amount + coin_amount)  = output / (lp_mint.supply + output) =>  output = coin_amount / total_coin_amount * lp_mint.supply
            let invariant_coin = InvariantPool {
                token_input: deduct_coin_amount,
                token_total: total_coin_without_take_pnl,
            };
            mint_lp_amount = invariant_coin
                .exchange_token_to_pool(amm.lp_amount, RoundDirection::Floor)
                .ok_or(AmmError::CalculationExRateFailure)?;
        } else {
            // base pc
            deduct_coin_amount = invariant
                .exchange_pc_to_coin(deposit.max_pc_amount, RoundDirection::Ceiling)
                .ok_or(AmmError::CalculationExRateFailure)?;
            deduct_pc_amount = deposit.max_pc_amount;
            if deduct_coin_amount > deposit.max_coin_amount {
                encode_ray_log(DepositLog {
                    log_type: LogType::Deposit.into_u8(),
                    max_coin: deposit.max_coin_amount,
                    max_pc: deposit.max_pc_amount,
                    base: deposit.base_side,
                    pool_coin: total_coin_without_take_pnl,
                    pool_pc: total_pc_without_take_pnl,
                    pool_lp: amm.lp_amount,
                    calc_pnl_x: target_orders.calc_pnl_x,
                    calc_pnl_y: target_orders.calc_pnl_y,
                    deduct_coin: deduct_coin_amount,
                    deduct_pc: deduct_pc_amount,
                    mint_lp: 0,
                });
                return Err(AmmError::ExceededSlippage.into());
            }
            let invariant_pc = InvariantPool {
                token_input: deduct_pc_amount,
                token_total: total_pc_without_take_pnl,
            };
            // pc_amount/ (total_pc_amount + pc_amount)  = output / (lp_mint.supply + output) =>  output = pc_amount / total_pc_amount * lp_mint.supply
            mint_lp_amount = invariant_pc
                .exchange_token_to_pool(amm.lp_amount, RoundDirection::Floor)
                .ok_or(AmmError::CalculationExRateFailure)?;
        }
        encode_ray_log(DepositLog {
            log_type: LogType::Deposit.into_u8(),
            max_coin: deposit.max_coin_amount,
            max_pc: deposit.max_pc_amount,
            base: deposit.base_side,
            pool_coin: total_coin_without_take_pnl,
            pool_pc: total_pc_without_take_pnl,
            pool_lp: amm.lp_amount,
            calc_pnl_x: target_orders.calc_pnl_x,
            calc_pnl_y: target_orders.calc_pnl_y,
            deduct_coin: deduct_coin_amount,
            deduct_pc: deduct_pc_amount,
            mint_lp: mint_lp_amount,
        });

        if deduct_coin_amount > user_source_coin.amount || deduct_pc_amount > user_source_pc.amount
        {
            return Err(AmmError::InsufficientFunds.into());
        }
        if mint_lp_amount == 0 || deduct_coin_amount == 0 || deduct_pc_amount == 0 {
            return Err(AmmError::InvalidInput.into());
        }

        Invokers::token_transfer(
            token_program_info.clone(),
            user_source_coin_info.clone(),
            amm_coin_vault_info.clone(),
            source_owner_info.clone(),
            deduct_coin_amount,
        )?;
        Invokers::token_transfer(
            token_program_info.clone(),
            user_source_pc_info.clone(),
            amm_pc_vault_info.clone(),
            source_owner_info.clone(),
            deduct_pc_amount,
        )?;
        Invokers::token_mint_to(
            token_program_info.clone(),
            amm_lp_mint_info.clone(),
            user_dest_lp_info.clone(),
            amm_authority_info.clone(),
            AUTHORITY_AMM,
            amm.nonce as u8,
            mint_lp_amount,
        )?;
        amm.lp_amount = amm.lp_amount.checked_add(mint_lp_amount).unwrap();

        target_orders.calc_pnl_x = x1
            .checked_add(Calculator::normalize_decimal_v2(
                deduct_pc_amount,
                amm.pc_decimals,
                amm.sys_decimal_value,
            ))
            .unwrap()
            .checked_sub(U128::from(delta_x))
            .unwrap()
            .as_u128();
        target_orders.calc_pnl_y = y1
            .checked_add(Calculator::normalize_decimal_v2(
                deduct_coin_amount,
                amm.coin_decimals,
                amm.sys_decimal_value,
            ))
            .unwrap()
            .checked_sub(U128::from(delta_y))
            .unwrap()
            .as_u128();
        Ok(())
    }

    pub fn process_withdrawpnl(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
        const ACCOUNT_LEN: usize = 17;
        let input_account_len = accounts.len();
        if input_account_len != ACCOUNT_LEN && input_account_len != ACCOUNT_LEN + 1 {
            return Err(AmmError::WrongAccountsNumber.into());
        }
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_config_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let user_pnl_coin_info = next_account_info(account_info_iter)?;
        let user_pnl_pc_info = next_account_info(account_info_iter)?;
        let pnl_owner_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer = next_account_info(account_info_iter)?;
        let mut referrer_pc_wallet = None;
        if input_account_len == ACCOUNT_LEN + 1 {
            referrer_pc_wallet = Some(next_account_info(account_info_iter)?);
            let referrer_pc_token =
                Self::unpack_token_account(&referrer_pc_wallet.unwrap(), token_program_info.key)?;
            check_assert_eq!(
                referrer_pc_token.owner,
                config_feature::referrer_pc_wallet::id(),
                "referrer_pc_owner",
                AmmError::InvalidOwner
            );
        }

        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        if amm_info.owner != program_id {
            return Err(AmmError::InvalidOwner.into());
        }

        let (pda, _) = Pubkey::find_program_address(&[&AMM_CONFIG_SEED], program_id);
        if pda != *amm_config_info.key || amm_config_info.owner != program_id {
            return Err(AmmError::InvalidConfigAccount.into());
        }
        let amm_config = AmmConfig::load_checked(&amm_config_info, program_id)?;

        if !pnl_owner_info.is_signer
            || (*pnl_owner_info.key != config_feature::amm_owner::ID
                && *pnl_owner_info.key != amm_config.pnl_owner)
        {
            return Err(AmmError::InvalidSignAccount.into());
        }
        // withdrawpnl in all status except Uninitialized
        if amm.status == AmmStatus::Uninitialized.into_u64() {
            msg!(&format!("withdrawpnl: status {}", amm.status));
            return Err(AmmError::InvalidStatus.into());
        }
        check_assert_eq!(
            *market_info.key,
            amm.market,
            "market",
            AmmError::InvalidMarket
        );
        check_assert_eq!(
            *amm_coin_vault_info.key,
            amm.coin_vault,
            "coin_vault",
            AmmError::InvalidCoinVault
        );
        check_assert_eq!(
            *amm_pc_vault_info.key,
            amm.pc_vault,
            "pc_vault",
            AmmError::InvalidPCVault
        );
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        check_assert_eq!(
            *market_program_info.key,
            amm.market_program,
            "market_program",
            AmmError::InvalidMarketProgram
        );
        check_assert_eq!(
            *amm_open_orders_info.key,
            amm.open_orders,
            "open_orders",
            AmmError::InvalidOpenOrders
        );
        check_assert_eq!(
            *amm_target_orders_info.key,
            amm.target_orders,
            "target_orders",
            AmmError::InvalidTargetOrders
        );
        let amm_coin_vault =
            Self::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
        let user_pnl_coin = Self::unpack_token_account(&user_pnl_coin_info, spl_token_program_id)?;
        let user_pnl_pc = Self::unpack_token_account(&user_pnl_pc_info, spl_token_program_id)?;
        let mut target_orders =
            TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
        if amm_coin_vault.mint != amm.coin_vault_mint || user_pnl_coin.mint != amm.coin_vault_mint {
            return Err(AmmError::InvalidCoinMint.into());
        }
        if amm_pc_vault.mint != amm.pc_vault_mint || user_pnl_pc.mint != amm.pc_vault_mint {
            return Err(AmmError::InvalidPCMint.into());
        }

        let (market_state, open_orders) = Self::load_serum_market_order(
            market_info,
            amm_open_orders_info,
            amm_authority_info,
            &amm,
            false,
        )?;

        msg!(arrform!(
            LOG_SIZE,
            "withdrawpnl need_take_coin:{}, need_take_pc:{}",
            amm.state_data.need_take_pnl_coin,
            amm.state_data.need_take_pnl_pc
        )
        .as_str());

        let (mut total_pc_without_take_pnl, mut total_coin_without_take_pnl) =
            Calculator::calc_total_without_take_pnl(
                amm_pc_vault.amount,
                amm_coin_vault.amount,
                &open_orders,
                &amm,
                &market_state,
                &market_event_queue_info,
                &amm_open_orders_info,
            )?;
        let x1 = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y1 = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        msg!(arrform!(
            LOG_SIZE,
            "withdrawpnl total_pc:{}, total_pc:{}, x:{}, y:{}",
            total_pc_without_take_pnl,
            total_coin_without_take_pnl,
            x1,
            y1
        )
        .as_str());

        // calc and update pnl
        let (delta_x, delta_y) = Self::calc_take_pnl(
            &target_orders,
            &mut amm,
            &mut total_pc_without_take_pnl,
            &mut total_coin_without_take_pnl,
            x1.as_u128().into(),
            y1.as_u128().into(),
        )?;
        msg!(arrform!(LOG_SIZE, "withdrawpnl total_pc:{}, total_pc:{}, delta_x:{}, delta_y:{}, need_take_coin:{}, need_take_pc:{}",total_pc_without_take_pnl, total_coin_without_take_pnl, delta_x, delta_y, amm.state_data.need_take_pnl_coin, amm.state_data.need_take_pnl_pc).as_str());

        if amm.state_data.need_take_pnl_coin <= amm_coin_vault.amount
            && amm.state_data.need_take_pnl_pc <= amm_pc_vault.amount
        {
            // coin & pc is enough, transfer directly
            Invokers::token_transfer_with_authority(
                token_program_info.clone(),
                amm_coin_vault_info.clone(),
                user_pnl_coin_info.clone(),
                amm_authority_info.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
                amm.state_data.need_take_pnl_coin,
            )?;
            Invokers::token_transfer_with_authority(
                token_program_info.clone(),
                amm_pc_vault_info.clone(),
                user_pnl_pc_info.clone(),
                amm_authority_info.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
                amm.state_data.need_take_pnl_pc,
            )?;
            // clear need take pnl
            amm.state_data.need_take_pnl_coin = 0u64;
            amm.state_data.need_take_pnl_pc = 0u64;
            // update target_orders.calc_pnl_x & target_orders.calc_pnl_y
            target_orders.calc_pnl_x = x1.checked_sub(U128::from(delta_x)).unwrap().as_u128();
            target_orders.calc_pnl_y = y1.checked_sub(U128::from(delta_y)).unwrap().as_u128();
        } else if amm.state_data.need_take_pnl_coin
            <= amm_coin_vault
                .amount
                .checked_add(open_orders.native_coin_free)
                .unwrap()
            && amm.state_data.need_take_pnl_pc
                <= amm_pc_vault
                    .amount
                    .checked_add(open_orders.native_pc_free)
                    .unwrap()
        {
            // settle & transfer
            Invokers::invoke_dex_settle_funds(
                market_program_info.clone(),
                market_info.clone(),
                amm_open_orders_info.clone(),
                amm_authority_info.clone(),
                market_coin_vault_info.clone(),
                market_pc_vault_info.clone(),
                amm_coin_vault_info.clone(),
                amm_pc_vault_info.clone(),
                market_vault_signer.clone(),
                token_program_info.clone(),
                referrer_pc_wallet.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
            )?;
            Invokers::token_transfer_with_authority(
                token_program_info.clone(),
                amm_coin_vault_info.clone(),
                user_pnl_coin_info.clone(),
                amm_authority_info.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
                amm.state_data.need_take_pnl_coin,
            )?;
            Invokers::token_transfer_with_authority(
                token_program_info.clone(),
                amm_pc_vault_info.clone(),
                user_pnl_pc_info.clone(),
                amm_authority_info.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
                amm.state_data.need_take_pnl_pc,
            )?;
            // clear need take pnl
            amm.state_data.need_take_pnl_coin = 0u64;
            amm.state_data.need_take_pnl_pc = 0u64;
            // update target_orders.calc_pnl_x & target_orders.calc_pnl_y
            target_orders.calc_pnl_x = x1.checked_sub(U128::from(delta_x)).unwrap().as_u128();
            target_orders.calc_pnl_y = y1.checked_sub(U128::from(delta_y)).unwrap().as_u128();
        } else if amm.state_data.need_take_pnl_coin
            >= amm_coin_vault
                .amount
                .checked_add(open_orders.native_coin_total)
                .unwrap()
            || amm.state_data.need_take_pnl_pc
                >= amm_pc_vault
                    .amount
                    .checked_add(open_orders.native_pc_total)
                    .unwrap()
        {
            // calc error need to disable
            amm.state = AmmState::CancelAllOrdersState.into_u64();
            amm.status = AmmStatus::Disabled.into_u64();
        }

        Ok(())
    }

    /// Processes an [Withdraw](enum.Instruction.html).
    pub fn process_withdraw(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        withdraw: WithdrawInstruction,
    ) -> ProgramResult {
        const ACCOUNT_LEN: usize = 20;
        let input_account_len = accounts.len();
        if input_account_len != ACCOUNT_LEN
            && input_account_len != ACCOUNT_LEN + 1
            && input_account_len != ACCOUNT_LEN + 2
            && input_account_len != ACCOUNT_LEN + 3
        {
            return Err(AmmError::WrongAccountsNumber.into());
        }
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_lp_mint_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        if input_account_len == ACCOUNT_LEN + 2 || input_account_len == ACCOUNT_LEN + 3 {
            let _padding_account_info1 = next_account_info(account_info_iter)?;
            let _padding_account_info2 = next_account_info(account_info_iter)?;
        }

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer = next_account_info(account_info_iter)?;

        let user_source_lp_info = next_account_info(account_info_iter)?;
        let user_dest_coin_info = next_account_info(account_info_iter)?;
        let user_dest_pc_info = next_account_info(account_info_iter)?;
        let source_lp_owner_info = next_account_info(account_info_iter)?;

        let market_event_q_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;

        let mut referrer_pc_wallet = None;
        if input_account_len == ACCOUNT_LEN + 1 || input_account_len == ACCOUNT_LEN + 3 {
            referrer_pc_wallet = Some(next_account_info(account_info_iter)?);
            if *referrer_pc_wallet.unwrap().key != Pubkey::default() {
                let referrer_pc_token = Self::unpack_token_account(
                    &referrer_pc_wallet.unwrap(),
                    token_program_info.key,
                )?;
                check_assert_eq!(
                    referrer_pc_token.owner,
                    config_feature::referrer_pc_wallet::id(),
                    "referrer_pc_owner",
                    AmmError::InvalidOwner
                );
            }
        }

        if referrer_pc_wallet.is_none() {
            referrer_pc_wallet = Some(amm_pc_vault_info);
        }
        if !source_lp_owner_info.is_signer {
            return Err(AmmError::InvalidSignAccount.into());
        }
        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        let mut target_orders =
            TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;

        if !AmmStatus::from_u64(amm.status).withdraw_permission() {
            return Err(AmmError::InvalidStatus.into());
        }
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        let enable_orderbook;
        if AmmStatus::from_u64(amm.status).orderbook_permission() {
            enable_orderbook = true;
        } else {
            enable_orderbook = false;
        }
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        // token_coin must be amm.coin_vault or token_dest_coin must not be amm.coin_vault
        if *amm_coin_vault_info.key != amm.coin_vault || *user_dest_coin_info.key == amm.coin_vault
        {
            return Err(AmmError::InvalidCoinVault.into());
        }
        // token_pc must be amm.pc_vault or token_dest_pc must not be amm.pc_vault
        if *amm_pc_vault_info.key != amm.pc_vault || *user_dest_pc_info.key == amm.pc_vault {
            return Err(AmmError::InvalidPCVault.into());
        }
        check_assert_eq!(
            *amm_target_orders_info.key,
            amm.target_orders,
            "target_orders",
            AmmError::InvalidTargetOrders
        );
        check_assert_eq!(
            *amm_lp_mint_info.key,
            amm.lp_mint,
            "lp_mint",
            AmmError::InvalidPoolMint
        );

        let amm_coin_vault =
            Self::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
        let user_dest_coin =
            Self::unpack_token_account(&user_dest_coin_info, spl_token_program_id)?;
        let user_dest_pc = Self::unpack_token_account(&user_dest_pc_info, spl_token_program_id)?;

        let lp_mint = Self::unpack_mint(&amm_lp_mint_info, spl_token_program_id)?;
        let user_source_lp =
            Self::unpack_token_account(&user_source_lp_info, spl_token_program_id)?;
        if user_source_lp.mint != *amm_lp_mint_info.key {
            return Err(AmmError::InvalidTokenLP.into());
        }
        if withdraw.amount > user_source_lp.amount {
            return Err(AmmError::InsufficientFunds.into());
        }
        if withdraw.amount > lp_mint.supply || withdraw.amount >= amm.lp_amount {
            return Err(AmmError::NotAllowZeroLP.into());
        }
        let (mut total_pc_without_take_pnl, mut total_coin_without_take_pnl) = if enable_orderbook {
            // check account
            check_assert_eq!(
                *market_info.key,
                amm.market,
                "market",
                AmmError::InvalidMarket
            );
            check_assert_eq!(
                *market_program_info.key,
                amm.market_program,
                "market_program",
                AmmError::InvalidMarketProgram
            );
            check_assert_eq!(
                *amm_open_orders_info.key,
                amm.open_orders,
                "open_orders",
                AmmError::InvalidOpenOrders
            );
            // load
            let (market_state, open_orders) = Self::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            let (bids, asks) = Self::get_amm_orders(&open_orders, bids_orders, asks_orders)?;
            // cancel all orders
            let mut amm_order_ids_vec = Vec::new();
            let mut order_ids = [0u64; 8];
            let mut count = 0;
            for i in 0..std::cmp::max(bids.len(), asks.len()) {
                if i < bids.len() {
                    order_ids[count] = bids[i].client_order_id();
                    count += 1;
                }
                if i < asks.len() {
                    order_ids[count] = asks[i].client_order_id();
                    count += 1;
                }
                if count == 8 {
                    amm_order_ids_vec.push(order_ids);
                    order_ids = [0u64; 8];
                    count = 0;
                }
            }
            if count != 0 {
                amm_order_ids_vec.push(order_ids);
            }
            for ids in amm_order_ids_vec.iter() {
                Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                    market_program_info.clone(),
                    market_info.clone(),
                    market_bids_info.clone(),
                    market_asks_info.clone(),
                    amm_open_orders_info.clone(),
                    amm_authority_info.clone(),
                    market_event_q_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    *ids,
                )?;
            }
            Invokers::invoke_dex_settle_funds(
                market_program_info.clone(),
                market_info.clone(),
                amm_open_orders_info.clone(),
                amm_authority_info.clone(),
                market_coin_vault_info.clone(),
                market_pc_vault_info.clone(),
                amm_coin_vault_info.clone(),
                amm_pc_vault_info.clone(),
                market_vault_signer.clone(),
                token_program_info.clone(),
                referrer_pc_wallet.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
            )?;

            if identity(market_state.coin_mint) != amm_coin_vault.mint.to_aligned_bytes()
                || identity(market_state.coin_mint) != user_dest_coin.mint.to_aligned_bytes()
            {
                return Err(AmmError::InvalidCoinMint.into());
            }
            if identity(market_state.pc_mint) != amm_pc_vault.mint.to_aligned_bytes()
                || identity(market_state.pc_mint) != user_dest_pc.mint.to_aligned_bytes()
            {
                return Err(AmmError::InvalidPCMint.into());
            }
            Calculator::calc_total_without_take_pnl(
                amm_pc_vault.amount,
                amm_coin_vault.amount,
                &open_orders,
                &amm,
                &market_state,
                &market_event_q_info,
                &amm_open_orders_info,
            )?
        } else {
            Calculator::calc_total_without_take_pnl_no_orderbook(
                amm_pc_vault.amount,
                amm_coin_vault.amount,
                &amm,
            )?
        };

        let x1 = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y1 = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );

        // calc and update pnl
        let mut delta_x: u128 = 0;
        let mut delta_y: u128 = 0;
        if amm.status != AmmStatus::WithdrawOnly.into_u64() {
            (delta_x, delta_y) = Self::calc_take_pnl(
                &target_orders,
                &mut amm,
                &mut total_pc_without_take_pnl,
                &mut total_coin_without_take_pnl,
                x1.as_u128().into(),
                y1.as_u128().into(),
            )?;
        }

        // coin_amount / total_coin_amount = amount / lp_mint.supply => coin_amount = total_coin_amount * amount / pool_mint.supply
        let invariant = InvariantPool {
            token_input: withdraw.amount,
            token_total: amm.lp_amount,
        };
        let coin_amount = invariant
            .exchange_pool_to_token(total_coin_without_take_pnl, RoundDirection::Floor)
            .ok_or(AmmError::CalculationExRateFailure)?;
        let pc_amount = invariant
            .exchange_pool_to_token(total_pc_without_take_pnl, RoundDirection::Floor)
            .ok_or(AmmError::CalculationExRateFailure)?;

        encode_ray_log(WithdrawLog {
            log_type: LogType::Withdraw.into_u8(),
            withdraw_lp: withdraw.amount,
            user_lp: user_source_lp.amount,
            pool_coin: total_coin_without_take_pnl,
            pool_pc: total_pc_without_take_pnl,
            pool_lp: amm.lp_amount,
            calc_pnl_x: target_orders.calc_pnl_x,
            calc_pnl_y: target_orders.calc_pnl_y,
            out_coin: coin_amount,
            out_pc: pc_amount,
        });
        if withdraw.amount == 0 || coin_amount == 0 || pc_amount == 0 {
            return Err(AmmError::InvalidInput.into());
        }

        if coin_amount < amm_coin_vault.amount && pc_amount < amm_pc_vault.amount {
            Invokers::token_transfer_with_authority(
                token_program_info.clone(),
                amm_coin_vault_info.clone(),
                user_dest_coin_info.clone(),
                amm_authority_info.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
                coin_amount,
            )?;
            Invokers::token_transfer_with_authority(
                token_program_info.clone(),
                amm_pc_vault_info.clone(),
                user_dest_pc_info.clone(),
                amm_authority_info.clone(),
                AUTHORITY_AMM,
                amm.nonce as u8,
                pc_amount,
            )?;
            Invokers::token_burn(
                token_program_info.clone(),
                user_source_lp_info.clone(),
                amm_lp_mint_info.clone(),
                source_lp_owner_info.clone(),
                withdraw.amount,
            )?;
            amm.lp_amount = amm.lp_amount.checked_sub(withdraw.amount).unwrap();
        } else {
            // calc error
            return Err(AmmError::TakePnlError.into());
        }

        // step4: update target_orders.calc_pnl_x & target_orders.calc_pnl_y
        target_orders.calc_pnl_x = x1
            .checked_sub(Calculator::normalize_decimal_v2(
                pc_amount,
                amm.pc_decimals,
                amm.sys_decimal_value,
            ))
            .unwrap()
            .checked_sub(U128::from(delta_x))
            .unwrap()
            .as_u128();
        target_orders.calc_pnl_y = y1
            .checked_sub(Calculator::normalize_decimal_v2(
                coin_amount,
                amm.coin_decimals,
                amm.sys_decimal_value,
            ))
            .unwrap()
            .checked_sub(U128::from(delta_y))
            .unwrap()
            .as_u128();
        Ok(())
    }

    pub fn process_swap_base_in(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        swap: SwapInstructionBaseIn,
    ) -> ProgramResult {
        const ACCOUNT_LEN: usize = 17;
        let input_account_len = accounts.len();
        if input_account_len != ACCOUNT_LEN && input_account_len != ACCOUNT_LEN + 1 {
            return Err(AmmError::WrongAccountsNumber.into());
        }
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        if input_account_len == ACCOUNT_LEN + 1 {
            let _amm_target_orders_info = next_account_info(account_info_iter)?;
        }
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;

        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        let enable_orderbook;
        if AmmStatus::from_u64(amm.status).orderbook_permission() {
            enable_orderbook = true;
        } else {
            enable_orderbook = false;
        }
        let market_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer = next_account_info(account_info_iter)?;

        let user_source_info = next_account_info(account_info_iter)?;
        let user_destination_info = next_account_info(account_info_iter)?;
        let user_source_owner = next_account_info(account_info_iter)?;
        if !user_source_owner.is_signer {
            return Err(AmmError::InvalidSignAccount.into());
        }
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        check_assert_eq!(
            *amm_coin_vault_info.key,
            amm.coin_vault,
            "coin_vault",
            AmmError::InvalidCoinVault
        );
        check_assert_eq!(
            *amm_pc_vault_info.key,
            amm.pc_vault,
            "pc_vault",
            AmmError::InvalidPCVault
        );

        if *user_source_info.key == amm.pc_vault || *user_source_info.key == amm.coin_vault {
            return Err(AmmError::InvalidUserToken.into());
        }
        if *user_destination_info.key == amm.pc_vault
            || *user_destination_info.key == amm.coin_vault
        {
            return Err(AmmError::InvalidUserToken.into());
        }

        let amm_coin_vault =
            Self::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;

        let user_source = Self::unpack_token_account(&user_source_info, spl_token_program_id)?;
        let user_destination =
            Self::unpack_token_account(&user_destination_info, spl_token_program_id)?;

        if !AmmStatus::from_u64(amm.status).swap_permission() {
            msg!(&format!("swap_base_in: status {}", amm.status));
            let clock = Clock::get()?;
            if amm.status == AmmStatus::OrderBookOnly.into_u64()
                && (clock.unix_timestamp as u64) >= amm.state_data.orderbook_to_init_time
            {
                amm.status = AmmStatus::Initialized.into_u64();
                msg!("swap_base_in: OrderBook to Initialized");
            } else {
                return Err(AmmError::InvalidStatus.into());
            }
        } else if amm.status == AmmStatus::WaitingTrade.into_u64() {
            let clock = Clock::get()?;
            if (clock.unix_timestamp as u64) < amm.state_data.pool_open_time {
                return Err(AmmError::InvalidStatus.into());
            } else {
                amm.status = AmmStatus::SwapOnly.into_u64();
                msg!("swap_base_in: WaitingTrade to SwapOnly");
            }
        }

        let total_pc_without_take_pnl;
        let total_coin_without_take_pnl;
        let mut bids: Vec<LeafNode> = Vec::new();
        let mut asks: Vec<LeafNode> = Vec::new();
        if enable_orderbook {
            check_assert_eq!(
                *amm_open_orders_info.key,
                amm.open_orders,
                "open_orders",
                AmmError::InvalidOpenOrders
            );
            check_assert_eq!(
                *market_program_info.key,
                amm.market_program,
                "market_program",
                AmmError::InvalidMarketProgram
            );
            check_assert_eq!(
                *market_info.key,
                amm.market,
                "market",
                AmmError::InvalidMarket
            );
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            (bids, asks) = Self::get_amm_orders(&open_orders, bids_orders, asks_orders)?;
            (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;
        } else {
            (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl_no_orderbook(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &amm,
                )?;
        }

        let swap_direction;
        if user_source.mint == amm_coin_vault.mint && user_destination.mint == amm_pc_vault.mint {
            swap_direction = SwapDirection::Coin2PC
        } else if user_source.mint == amm_pc_vault.mint
            && user_destination.mint == amm_coin_vault.mint
        {
            swap_direction = SwapDirection::PC2Coin
        } else {
            return Err(AmmError::InvalidUserToken.into());
        }
        if user_source.amount < swap.amount_in {
            encode_ray_log(SwapBaseInLog {
                log_type: LogType::SwapBaseIn.into_u8(),
                amount_in: swap.amount_in,
                minimum_out: swap.minimum_amount_out,
                direction: swap_direction as u64,
                user_source: user_source.amount,
                pool_coin: total_coin_without_take_pnl,
                pool_pc: total_pc_without_take_pnl,
                out_amount: 0,
            });
            return Err(AmmError::InsufficientFunds.into());
        }
        let swap_fee = U128::from(swap.amount_in)
            .checked_mul(amm.fees.swap_fee_numerator.into())
            .unwrap()
            .checked_ceil_div(amm.fees.swap_fee_denominator.into())
            .unwrap()
            .0;
        let swap_in_after_deduct_fee = U128::from(swap.amount_in).checked_sub(swap_fee).unwrap();
        let swap_amount_out = Calculator::swap_token_amount_base_in(
            swap_in_after_deduct_fee,
            total_pc_without_take_pnl.into(),
            total_coin_without_take_pnl.into(),
            swap_direction,
        )
        .as_u64();
        encode_ray_log(SwapBaseInLog {
            log_type: LogType::SwapBaseIn.into_u8(),
            amount_in: swap.amount_in,
            minimum_out: swap.minimum_amount_out,
            direction: swap_direction as u64,
            user_source: user_source.amount,
            pool_coin: total_coin_without_take_pnl,
            pool_pc: total_pc_without_take_pnl,
            out_amount: swap_amount_out,
        });
        if swap_amount_out < swap.minimum_amount_out {
            return Err(AmmError::ExceededSlippage.into());
        }
        if swap_amount_out == 0 || swap.amount_in == 0 {
            return Err(AmmError::InvalidInput.into());
        }

        match swap_direction {
            SwapDirection::Coin2PC => {
                if swap_amount_out >= total_pc_without_take_pnl {
                    return Err(AmmError::InsufficientFunds.into());
                }

                if enable_orderbook {
                    // coin -> pc, need cancel buy order
                    if !bids.is_empty() {
                        let mut amm_order_ids_vec = Vec::new();
                        let mut order_ids = [0u64; 8];
                        let mut count = 0;
                        // fetch cancel order ids{
                        for order in bids.into_iter() {
                            order_ids[count] = order.client_order_id();
                            count += 1;
                            if count == 8 {
                                amm_order_ids_vec.push(order_ids);
                                order_ids = [0u64; 8];
                                count = 0;
                            }
                        }
                        if count != 0 {
                            amm_order_ids_vec.push(order_ids);
                        }
                        for ids in amm_order_ids_vec.iter() {
                            Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                                market_program_info.clone(),
                                market_info.clone(),
                                market_bids_info.clone(),
                                market_asks_info.clone(),
                                amm_open_orders_info.clone(),
                                amm_authority_info.clone(),
                                market_event_queue_info.clone(),
                                AUTHORITY_AMM,
                                amm.nonce as u8,
                                *ids,
                            )?;
                        }
                    }

                    if swap_amount_out > amm_pc_vault.amount {
                        // need settle funds
                        Invokers::invoke_dex_settle_funds(
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_coin_vault_info.clone(),
                            market_pc_vault_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            market_vault_signer.clone(),
                            token_program_info.clone(),
                            Some(&amm_pc_vault_info.clone()),
                            AUTHORITY_AMM,
                            amm.nonce as u8,
                        )?;
                    }
                }
                // deposit source coin to amm_coin_vault
                Invokers::token_transfer(
                    token_program_info.clone(),
                    user_source_info.clone(),
                    amm_coin_vault_info.clone(),
                    user_source_owner.clone(),
                    swap.amount_in,
                )?;
                // withdraw amm_pc_vault to destination pc
                Invokers::token_transfer_with_authority(
                    token_program_info.clone(),
                    amm_pc_vault_info.clone(),
                    user_destination_info.clone(),
                    amm_authority_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    swap_amount_out,
                )?;
                // update state_data data
                amm.state_data.swap_coin_in_amount = amm
                    .state_data
                    .swap_coin_in_amount
                    .checked_add(swap.amount_in.into())
                    .unwrap();
                amm.state_data.swap_pc_out_amount = amm
                    .state_data
                    .swap_pc_out_amount
                    .checked_add(swap_amount_out.into())
                    .unwrap();
                // charge coin as swap fee
                amm.state_data.swap_acc_coin_fee = amm
                    .state_data
                    .swap_acc_coin_fee
                    .checked_add(swap_fee.as_u64())
                    .unwrap();
            }
            SwapDirection::PC2Coin => {
                if swap_amount_out >= total_coin_without_take_pnl {
                    return Err(AmmError::InsufficientFunds.into());
                }

                if enable_orderbook {
                    // pc -> coin, need cancel sell order
                    if !asks.is_empty() {
                        let mut amm_order_ids_vec = Vec::new();
                        let mut order_ids = [0u64; 8];
                        let mut count = 0;
                        // fetch cancel order ids{
                        for order in asks.into_iter() {
                            order_ids[count] = order.client_order_id();
                            count += 1;
                            if count == 8 {
                                amm_order_ids_vec.push(order_ids);
                                order_ids = [0u64; 8];
                                count = 0;
                            }
                        }
                        if count != 0 {
                            amm_order_ids_vec.push(order_ids);
                        }
                        for ids in amm_order_ids_vec.iter() {
                            Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                                market_program_info.clone(),
                                market_info.clone(),
                                market_bids_info.clone(),
                                market_asks_info.clone(),
                                amm_open_orders_info.clone(),
                                amm_authority_info.clone(),
                                market_event_queue_info.clone(),
                                AUTHORITY_AMM,
                                amm.nonce as u8,
                                *ids,
                            )?;
                        }
                    }

                    if swap_amount_out > amm_coin_vault.amount {
                        Invokers::invoke_dex_settle_funds(
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_coin_vault_info.clone(),
                            market_pc_vault_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            market_vault_signer.clone(),
                            token_program_info.clone(),
                            Some(&amm_pc_vault_info.clone()),
                            AUTHORITY_AMM,
                            amm.nonce as u8,
                        )?;
                    }
                }
                // deposit source pc to amm_pc_vault
                Invokers::token_transfer(
                    token_program_info.clone(),
                    user_source_info.clone(),
                    amm_pc_vault_info.clone(),
                    user_source_owner.clone(),
                    swap.amount_in,
                )?;
                // withdraw amm_coin_vault to destination coin
                Invokers::token_transfer_with_authority(
                    token_program_info.clone(),
                    amm_coin_vault_info.clone(),
                    user_destination_info.clone(),
                    amm_authority_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    swap_amount_out,
                )?;
                // update state_data data
                amm.state_data.swap_pc_in_amount = amm
                    .state_data
                    .swap_pc_in_amount
                    .checked_add(swap.amount_in.into())
                    .unwrap();
                amm.state_data.swap_coin_out_amount = amm
                    .state_data
                    .swap_coin_out_amount
                    .checked_add(swap_amount_out.into())
                    .unwrap();
                // charge pc as swap fee
                amm.state_data.swap_acc_pc_fee = amm
                    .state_data
                    .swap_acc_pc_fee
                    .checked_add(swap_fee.as_u64())
                    .unwrap();
            }
        };

        Ok(())
    }

    pub fn process_swap_base_out(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        swap: SwapInstructionBaseOut,
    ) -> ProgramResult {
        const SWAP_ACCOUNT_NUM: usize = 17;
        let input_account_len = accounts.len();
        if input_account_len != SWAP_ACCOUNT_NUM && input_account_len != SWAP_ACCOUNT_NUM + 1 {
            return Err(AmmError::WrongAccountsNumber.into());
        }
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        if input_account_len == SWAP_ACCOUNT_NUM + 1 {
            let _amm_target_orders_info = next_account_info(account_info_iter)?;
        }
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;

        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        let enable_orderbook;
        if AmmStatus::from_u64(amm.status).orderbook_permission() {
            enable_orderbook = true;
        } else {
            enable_orderbook = false;
        }

        let market_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer = next_account_info(account_info_iter)?;

        let user_source_info = next_account_info(account_info_iter)?;
        let user_destination_info = next_account_info(account_info_iter)?;
        let user_source_owner = next_account_info(account_info_iter)?;
        if !user_source_owner.is_signer {
            return Err(AmmError::InvalidSignAccount.into());
        }

        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        let authority = Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?;
        check_assert_eq!(
            *amm_authority_info.key,
            authority,
            "authority",
            AmmError::InvalidProgramAddress
        );
        check_assert_eq!(
            *amm_coin_vault_info.key,
            amm.coin_vault,
            "coin_vault",
            AmmError::InvalidCoinVault
        );
        check_assert_eq!(
            *amm_pc_vault_info.key,
            amm.pc_vault,
            "pc_vault",
            AmmError::InvalidPCVault
        );

        if *user_source_info.key == amm.pc_vault || *user_source_info.key == amm.coin_vault {
            return Err(AmmError::InvalidUserToken.into());
        }
        if *user_destination_info.key == amm.pc_vault
            || *user_destination_info.key == amm.coin_vault
        {
            return Err(AmmError::InvalidUserToken.into());
        }

        let amm_coin_vault =
            Self::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;

        let user_source = Self::unpack_token_account(&user_source_info, spl_token_program_id)?;
        let user_destination =
            Self::unpack_token_account(&user_destination_info, spl_token_program_id)?;

        if !AmmStatus::from_u64(amm.status).swap_permission() {
            msg!(&format!("swap_base_out: status {}", amm.status));
            let clock = Clock::get()?;
            if amm.status == AmmStatus::OrderBookOnly.into_u64()
                && (clock.unix_timestamp as u64) >= amm.state_data.orderbook_to_init_time
            {
                amm.status = AmmStatus::Initialized.into_u64();
                msg!("swap_base_out: OrderBook to Initialized");
            } else {
                return Err(AmmError::InvalidStatus.into());
            }
        } else if amm.status == AmmStatus::WaitingTrade.into_u64() {
            let clock = Clock::get()?;
            if (clock.unix_timestamp as u64) < amm.state_data.pool_open_time {
                return Err(AmmError::InvalidStatus.into());
            } else {
                amm.status = AmmStatus::SwapOnly.into_u64();
                msg!("swap_base_out: WaitingTrade to SwapOnly");
            }
        }

        let total_pc_without_take_pnl;
        let total_coin_without_take_pnl;
        let mut bids: Vec<LeafNode> = Vec::new();
        let mut asks: Vec<LeafNode> = Vec::new();
        if enable_orderbook {
            check_assert_eq!(
                *amm_open_orders_info.key,
                amm.open_orders,
                "open_orders",
                AmmError::InvalidOpenOrders
            );
            check_assert_eq!(
                *market_program_info.key,
                amm.market_program,
                "market_program",
                AmmError::InvalidMarketProgram
            );
            check_assert_eq!(
                *market_info.key,
                amm.market,
                "market",
                AmmError::InvalidMarket
            );
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            (bids, asks) = Self::get_amm_orders(&open_orders, bids_orders, asks_orders)?;
            (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;
        } else {
            (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl_no_orderbook(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &amm,
                )?;
        }

        let swap_direction;
        if user_source.mint == amm_coin_vault.mint && user_destination.mint == amm_pc_vault.mint {
            swap_direction = SwapDirection::Coin2PC
        } else if user_source.mint == amm_pc_vault.mint
            && user_destination.mint == amm_coin_vault.mint
        {
            swap_direction = SwapDirection::PC2Coin
        } else {
            return Err(AmmError::InvalidUserToken.into());
        }

        let swap_in_before_add_fee = Calculator::swap_token_amount_base_out(
            swap.amount_out.into(),
            total_pc_without_take_pnl.into(),
            total_coin_without_take_pnl.into(),
            swap_direction,
        );
        // swap_in_after_add_fee * (1 - 0.0025) = swap_in_before_add_fee
        // swap_in_after_add_fee = swap_in_before_add_fee / (1 - 0.0025)
        let swap_in_after_add_fee = swap_in_before_add_fee
            .checked_mul(amm.fees.swap_fee_denominator.into())
            .unwrap()
            .checked_ceil_div(
                (amm.fees
                    .swap_fee_denominator
                    .checked_sub(amm.fees.swap_fee_numerator)
                    .unwrap())
                .into(),
            )
            .unwrap()
            .0
            .as_u64();
        let swap_fee = swap_in_after_add_fee
            .checked_sub(swap_in_before_add_fee.as_u64())
            .unwrap();
        encode_ray_log(SwapBaseOutLog {
            log_type: LogType::SwapBaseOut.into_u8(),
            max_in: swap.max_amount_in,
            amount_out: swap.amount_out,
            direction: swap_direction as u64,
            user_source: user_source.amount,
            pool_coin: total_coin_without_take_pnl,
            pool_pc: total_pc_without_take_pnl,
            deduct_in: swap_in_after_add_fee,
        });
        if user_source.amount < swap_in_after_add_fee {
            return Err(AmmError::InsufficientFunds.into());
        }
        if swap.max_amount_in < swap_in_after_add_fee {
            return Err(AmmError::ExceededSlippage.into());
        }
        if swap_in_after_add_fee == 0 || swap.amount_out == 0 {
            return Err(AmmError::InvalidInput.into());
        }

        match swap_direction {
            SwapDirection::Coin2PC => {
                if swap.amount_out >= total_pc_without_take_pnl {
                    return Err(AmmError::InsufficientFunds.into());
                }

                if enable_orderbook {
                    // coin -> pc, need cancel buy order
                    if !bids.is_empty() {
                        let mut amm_order_ids_vec = Vec::new();
                        let mut order_ids = [0u64; 8];
                        let mut count = 0;
                        // fetch cancel order ids
                        for order in bids.into_iter() {
                            order_ids[count] = order.client_order_id();
                            count += 1;
                            if count == 8 {
                                amm_order_ids_vec.push(order_ids);
                                order_ids = [0u64; 8];
                                count = 0;
                            }
                        }
                        if count != 0 {
                            amm_order_ids_vec.push(order_ids);
                        }
                        for ids in amm_order_ids_vec.iter() {
                            Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                                market_program_info.clone(),
                                market_info.clone(),
                                market_bids_info.clone(),
                                market_asks_info.clone(),
                                amm_open_orders_info.clone(),
                                amm_authority_info.clone(),
                                market_event_queue_info.clone(),
                                AUTHORITY_AMM,
                                amm.nonce as u8,
                                *ids,
                            )?;
                        }
                    }
                    if swap.amount_out > amm_pc_vault.amount {
                        // need settle funds
                        Invokers::invoke_dex_settle_funds(
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_coin_vault_info.clone(),
                            market_pc_vault_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            market_vault_signer.clone(),
                            token_program_info.clone(),
                            Some(&amm_pc_vault_info.clone()),
                            AUTHORITY_AMM,
                            amm.nonce as u8,
                        )?;
                    }
                }
                // deposit source coin to amm_coin_vault
                Invokers::token_transfer(
                    token_program_info.clone(),
                    user_source_info.clone(),
                    amm_coin_vault_info.clone(),
                    user_source_owner.clone(),
                    swap_in_after_add_fee,
                )?;
                // withdraw amm_pc_vault to destination pc
                Invokers::token_transfer_with_authority(
                    token_program_info.clone(),
                    amm_pc_vault_info.clone(),
                    user_destination_info.clone(),
                    amm_authority_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    swap.amount_out,
                )?;
                // update state_data data
                amm.state_data.swap_coin_in_amount = amm
                    .state_data
                    .swap_coin_in_amount
                    .checked_add(swap_in_after_add_fee.into())
                    .unwrap();
                amm.state_data.swap_pc_out_amount = amm
                    .state_data
                    .swap_pc_out_amount
                    .checked_add(Calculator::to_u128(swap.amount_out)?)
                    .unwrap();
                // charge coin as swap fee
                amm.state_data.swap_acc_coin_fee = amm
                    .state_data
                    .swap_acc_coin_fee
                    .checked_add(swap_fee)
                    .unwrap();
            }
            SwapDirection::PC2Coin => {
                if swap.amount_out >= total_coin_without_take_pnl {
                    return Err(AmmError::InsufficientFunds.into());
                }

                if enable_orderbook {
                    // pc -> coin, need cancel sell order
                    if !asks.is_empty() {
                        let mut amm_order_ids_vec = Vec::new();
                        let mut order_ids = [0u64; 8];
                        let mut count = 0;
                        // fetch cancel order ids
                        for order in asks.into_iter() {
                            order_ids[count] = order.client_order_id();
                            count += 1;
                            if count == 8 {
                                amm_order_ids_vec.push(order_ids);
                                order_ids = [0u64; 8];
                                count = 0;
                            }
                        }
                        if count != 0 {
                            amm_order_ids_vec.push(order_ids);
                        }
                        for ids in amm_order_ids_vec.iter() {
                            Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                                market_program_info.clone(),
                                market_info.clone(),
                                market_bids_info.clone(),
                                market_asks_info.clone(),
                                amm_open_orders_info.clone(),
                                amm_authority_info.clone(),
                                market_event_queue_info.clone(),
                                AUTHORITY_AMM,
                                amm.nonce as u8,
                                *ids,
                            )?;
                        }
                    }
                    if swap.amount_out > amm_coin_vault.amount {
                        Invokers::invoke_dex_settle_funds(
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_asks_info.clone(),
                            market_pc_vault_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            market_vault_signer.clone(),
                            token_program_info.clone(),
                            Some(&amm_pc_vault_info.clone()),
                            AUTHORITY_AMM,
                            amm.nonce as u8,
                        )?;
                    }
                }

                // deposit source pc to amm_pc_vault
                Invokers::token_transfer(
                    token_program_info.clone(),
                    user_source_info.clone(),
                    amm_pc_vault_info.clone(),
                    user_source_owner.clone(),
                    swap_in_after_add_fee,
                )?;
                // withdraw amm_coin_vault to destination coin
                Invokers::token_transfer_with_authority(
                    token_program_info.clone(),
                    amm_coin_vault_info.clone(),
                    user_destination_info.clone(),
                    amm_authority_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    swap.amount_out,
                )?;
                // update state_data data
                amm.state_data.swap_pc_in_amount = amm
                    .state_data
                    .swap_pc_in_amount
                    .checked_add(swap_in_after_add_fee.into())
                    .unwrap();
                amm.state_data.swap_coin_out_amount = amm
                    .state_data
                    .swap_coin_out_amount
                    .checked_add(swap.amount_out.into())
                    .unwrap();
                // charge pc as swap fee
                amm.state_data.swap_acc_pc_fee = amm
                    .state_data
                    .swap_acc_pc_fee
                    .checked_add(swap_fee)
                    .unwrap();
            }
        };

        Ok(())
    }

    pub fn process_migrate_to_openbook(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;
        let system_program_info = next_account_info(account_info_iter)?;
        let rent_sysvar_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer = next_account_info(account_info_iter)?;

        let new_amm_open_orders_info = next_account_info(account_info_iter)?;
        let new_market_program_info = next_account_info(account_info_iter)?;
        let new_market_info = next_account_info(account_info_iter)?;
        let admin_info = next_account_info(account_info_iter)?;
        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        if !admin_info.is_signer || *admin_info.key != config_feature::amm_owner::ID {
            return Err(AmmError::InvalidSignAccount.into());
        }
        let authority = Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?;
        check_assert_eq!(
            *amm_authority_info.key,
            authority,
            "authority",
            AmmError::InvalidProgramAddress
        );
        let spl_token_program_id = token_program_info.key;
        check_assert_eq!(
            *spl_token_program_id,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        check_assert_eq!(
            *system_program_info.key,
            solana_program::system_program::id(),
            "sys_program",
            AmmError::InvalidSysProgramAddress
        );
        // old account check
        check_assert_eq!(
            *amm_coin_vault_info.key,
            amm.coin_vault,
            "coin_vault",
            AmmError::InvalidCoinVault
        );
        check_assert_eq!(
            *amm_pc_vault_info.key,
            amm.pc_vault,
            "pc_vault",
            AmmError::InvalidPCVault
        );
        check_assert_eq!(
            *amm_target_orders_info.key,
            amm.target_orders,
            "target_orders",
            AmmError::InvalidPCVault
        );
        check_assert_eq!(
            *market_info.key,
            amm.market,
            "market",
            AmmError::InvalidMarket
        );
        check_assert_eq!(
            *market_program_info.key,
            amm.market_program,
            "market_program",
            AmmError::InvalidMarketProgram
        );
        check_assert_eq!(
            *amm_open_orders_info.key,
            amm.open_orders,
            "open_orders",
            AmmError::InvalidOpenOrders
        );
        let mut target =
            TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
        let pnl_pc_amount = Calculator::restore_decimal(
            target.calc_pnl_x.into(),
            amm.pc_decimals,
            amm.sys_decimal_value,
        )
        .as_u64();
        let pnl_coin_amount = Calculator::restore_decimal(
            target.calc_pnl_y.into(),
            amm.coin_decimals,
            amm.sys_decimal_value,
        )
        .as_u64();
        // cancel amm orders in old market
        Self::do_cancel_amm_orders(
            &amm,
            amm_authority_info,
            amm_open_orders_info,
            market_program_info,
            market_info,
            market_bids_info,
            market_asks_info,
            market_event_queue_info,
            AUTHORITY_AMM,
        )?;
        // settle assets
        Invokers::invoke_dex_settle_funds(
            market_program_info.clone(),
            market_info.clone(),
            amm_open_orders_info.clone(),
            amm_authority_info.clone(),
            market_coin_vault_info.clone(),
            market_pc_vault_info.clone(),
            amm_coin_vault_info.clone(),
            amm_pc_vault_info.clone(),
            market_vault_signer.clone(),
            token_program_info.clone(),
            None,
            AUTHORITY_AMM,
            amm.nonce as u8,
        )?;
        let (market_state, open_orders) = Self::load_serum_market_order(
            market_info,
            amm_open_orders_info,
            amm_authority_info,
            &amm,
            false,
        )?;
        if identity(market_state.coin_mint) != amm.coin_vault_mint.to_aligned_bytes() {
            return Err(AmmError::InvalidCoinMint.into());
        }
        if identity(market_state.pc_mint) != amm.pc_vault_mint.to_aligned_bytes() {
            return Err(AmmError::InvalidPCMint.into());
        }
        // Partial filled without consumed assets exclude native total in open_orders will not be settled back to amm.
        // It is necessary to check that all assets have been indeed settled back to amm
        let (pc_total_in_serum, coin_total_in_serum) = Calculator::calc_exact_vault_in_serum(
            &open_orders,
            &market_state,
            market_event_queue_info,
            amm_open_orders_info,
        )?;
        if pc_total_in_serum != 0 || coin_total_in_serum != 0 {
            // Invokers::invoke_dex_close_open_orders(
            //     serum_dex_info.clone(),
            //     amm_open_orders_info.clone(),
            //     authority_info.clone(),
            //     admin_info.clone(),
            //     market_info.clone(),
            //     AUTHORITY_AMM,
            //     amm.nonce as u8,
            // )?;
            msg!(
                "{}, {}, {}, {}, {:?}",
                pc_total_in_serum,
                coin_total_in_serum,
                identity(open_orders.native_pc_total),
                identity(open_orders.native_coin_total),
                identity(open_orders.free_slot_bits)
            );
            return Err(AmmError::UnknownAmmError.into());
        }
        // check new dex
        check_assert_eq!(
            *new_market_program_info.key,
            config_feature::openbook_program::id(),
            "new_market_program",
            AmmError::InvalidMarketProgram
        );
        let new_market_pc_lot_size;
        let new_market_coin_lot_size;
        {
            let new_market_state = Market::load(
                new_market_info,
                &config_feature::openbook_program::id(),
                false,
            )?;
            new_market_pc_lot_size = new_market_state.pc_lot_size;
            new_market_coin_lot_size = new_market_state.coin_lot_size;
            if identity(new_market_state.coin_mint) != amm.coin_vault_mint.to_aligned_bytes() {
                return Err(AmmError::InvalidCoinMint.into());
            }
            if identity(new_market_state.pc_mint) != amm.pc_vault_mint.to_aligned_bytes() {
                return Err(AmmError::InvalidPCMint.into());
            }
        }
        // create amm open order account
        Self::generate_amm_associated_account(
            program_id,
            new_market_program_info.key,
            new_market_info,
            new_amm_open_orders_info,
            admin_info,
            system_program_info,
            rent_sysvar_info,
            OPEN_ORDER_ASSOCIATED_SEED,
            size_of::<serum_dex::state::OpenOrders>() + 12,
        )?;
        // init open orders account
        Invokers::invoke_dex_init_open_orders(
            new_market_program_info.clone(),
            new_amm_open_orders_info.clone(),
            amm_authority_info.clone(),
            new_market_info.clone(),
            rent_sysvar_info.clone(),
            AUTHORITY_AMM,
            amm.nonce as u8,
        )?;
        // update pool
        amm.market_program = *new_market_program_info.key;
        amm.market = *new_market_info.key;
        amm.open_orders = *new_amm_open_orders_info.key;
        let nonce = amm.nonce as u8;
        let pool_open_time = amm.state_data.pool_open_time;
        let coin_decimals = amm.coin_decimals as u8;
        let pc_decimals = amm.pc_decimals as u8;
        amm.initialize(
            nonce,
            pool_open_time,
            coin_decimals,
            pc_decimals,
            new_market_coin_lot_size,
            new_market_pc_lot_size,
        )?;
        amm.status = AmmStatus::WaitingTrade.into_u64();
        amm.reset_flag = AmmResetFlag::ResetYes.into_u64();

        if new_market_coin_lot_size != market_state.coin_lot_size
            || new_market_pc_lot_size != market_state.pc_lot_size
        {
            // market lot size is different, calc_pnl_x & calc_pnl_x must update
            target.calc_pnl_x = Calculator::normalize_decimal_v2(
                pnl_pc_amount,
                amm.pc_decimals,
                amm.sys_decimal_value,
            )
            .as_u128();
            target.calc_pnl_y = Calculator::normalize_decimal_v2(
                pnl_coin_amount,
                amm.coin_decimals,
                amm.sys_decimal_value,
            )
            .as_u128();
        }

        Ok(())
    }

    /// withdraw_srm
    pub fn process_withdraw_srm(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        withdrawsrm: WithdrawSrmInstruction,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_owner_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let srm_token_info = next_account_info(account_info_iter)?;
        let dest_srm_token_info = next_account_info(account_info_iter)?;

        msg!("withdraw_srm: {}", withdrawsrm.amount);
        let amm = AmmInfo::load_checked(&amm_info, program_id)?;
        if amm.status == AmmStatus::Uninitialized.into_u64() {
            msg!(&format!("withdraw_srm: status {}", amm.status));
            return Err(AmmError::InvalidStatus.into());
        }
        if !amm_owner_info.is_signer || *amm_owner_info.key != config_feature::amm_owner::ID {
            return Err(AmmError::InvalidSignAccount.into());
        }
        // check_assert_eq!(
        //     *amm_owner_info.key,
        //     amm.amm_owner,
        //     "amm_owner",
        //     AmmError::InvalidOwner
        // );
        check_assert_eq!(
            *token_program_info.key,
            spl_token::id(),
            "spl_token_program",
            AmmError::InvalidSplTokenProgram
        );
        let spl_token_program_id = token_program_info.key;
        let authority = Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?;
        check_assert_eq!(
            *amm_authority_info.key,
            authority,
            "authority",
            AmmError::InvalidProgramAddress
        );
        let srm_token = Self::unpack_token_account(&srm_token_info, spl_token_program_id)?;
        check_assert_eq!(
            srm_token.owner,
            *amm_authority_info.key,
            "srm_token_owner",
            AmmError::InvalidOwner
        );
        if srm_token.mint != srm_token::id() && srm_token.mint != msrm_token::id() {
            return Err(AmmError::InvalidSrmMint.into());
        }
        let dest_srm_token =
            Self::unpack_token_account(&dest_srm_token_info, spl_token_program_id)?;
        check_assert_eq!(
            srm_token.mint,
            dest_srm_token.mint,
            "srm_token_mint",
            AmmError::InvalidInput
        );
        if withdrawsrm.amount > srm_token.amount {
            return Err(AmmError::InsufficientFunds.into());
        }

        Invokers::token_transfer_with_authority(
            token_program_info.clone(),
            srm_token_info.clone(),
            dest_srm_token_info.clone(),
            amm_authority_info.clone(),
            AUTHORITY_AMM,
            amm.nonce as u8,
            withdrawsrm.amount,
        )?;

        Ok(())
    }

    fn simulate_pool_info(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
    ) -> Result<GetPoolData, ProgramError> {
        const ACCOUNT_LEN: usize = 8;
        let input_account_len = accounts.len();
        let account_info_iter = &mut accounts.iter();
        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let amm_lp_mint_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;

        let amm = AmmInfo::load_checked(&amm_info, program_id)?;
        let authority = Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?;
        Self::check_account_readonly(amm_info)?;
        Self::check_account_readonly(amm_open_orders_info)?;
        Self::check_account_readonly(amm_coin_vault_info)?;
        Self::check_account_readonly(amm_pc_vault_info)?;
        Self::check_account_readonly(amm_lp_mint_info)?;
        Self::check_account_readonly(market_info)?;
        Self::check_account_readonly(market_event_queue_info)?;
        check_assert_eq!(
            *amm_authority_info.key,
            authority,
            "authority",
            AmmError::InvalidProgramAddress
        );
        check_assert_eq!(
            *amm_coin_vault_info.key,
            amm.coin_vault,
            "coin_vault",
            AmmError::InvalidCoinVault
        );
        check_assert_eq!(
            *amm_pc_vault_info.key,
            amm.pc_vault,
            "pc_vault",
            AmmError::InvalidPCVault
        );
        check_assert_eq!(
            *amm_lp_mint_info.key,
            amm.lp_mint,
            "lp_mint",
            AmmError::InvalidPoolMint
        );
        check_assert_eq!(
            *amm_open_orders_info.key,
            amm.open_orders,
            "open_orders",
            AmmError::InvalidOpenOrders
        );
        check_assert_eq!(
            *market_info.key,
            amm.market,
            "market",
            AmmError::InvalidMarket
        );
        let pnl_pc_amount;
        let pnl_coin_amount;
        if input_account_len == ACCOUNT_LEN + 1 {
            let target_orders_info = next_account_info(account_info_iter)?;
            check_assert_eq!(
                *target_orders_info.key,
                amm.target_orders,
                "target_orders",
                AmmError::InvalidTargetOrders
            );
            let target = TargetOrders::load_checked(&target_orders_info, program_id, amm_info.key)?;
            pnl_pc_amount = Calculator::restore_decimal(
                target.calc_pnl_x.into(),
                amm.pc_decimals,
                amm.sys_decimal_value,
            )
            .as_u64();
            pnl_coin_amount = Calculator::restore_decimal(
                target.calc_pnl_y.into(),
                amm.coin_decimals,
                amm.sys_decimal_value,
            )
            .as_u64();
        } else {
            pnl_pc_amount = 0;
            pnl_coin_amount = 0;
        }

        let amm_coin_vault = Self::unpack_token_account(&amm_coin_vault_info, &spl_token::id())?;
        let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, &spl_token::id())?;
        let lp_mint = Self::unpack_mint(&amm_lp_mint_info, &spl_token::id())?;
        let (market_state, open_orders) = Self::load_serum_market_order(
            market_info,
            amm_open_orders_info,
            amm_authority_info,
            &amm,
            false,
        )?;
        if identity(market_state.coin_mint) != amm_coin_vault.mint.to_aligned_bytes() {
            return Err(AmmError::InvalidCoinMint.into());
        }
        if identity(market_state.pc_mint) != amm_pc_vault.mint.to_aligned_bytes() {
            return Err(AmmError::InvalidPCMint.into());
        }
        let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
            Calculator::calc_total_without_take_pnl(
                amm_pc_vault.amount,
                amm_coin_vault.amount,
                &open_orders,
                &amm,
                &market_state,
                &market_event_queue_info,
                &amm_open_orders_info,
            )?;
        let pool_info_data = GetPoolData {
            status: amm.status,
            coin_decimals: amm.coin_decimals,
            pc_decimals: amm.pc_decimals,
            lp_decimals: lp_mint.decimals.into(),
            pool_pc_amount: total_pc_without_take_pnl,
            pool_coin_amount: total_coin_without_take_pnl,
            pnl_pc_amount,
            pnl_coin_amount,
            pool_lp_supply: amm.lp_amount,
            pool_open_time: amm.state_data.pool_open_time,
            amm_id: amm_info.key.to_string(),
        };
        return Ok(pool_info_data);
    }

    fn simulate_swap_base_in(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        simulate: SimulateInstruction,
    ) -> Result<GetSwapBaseInData, ProgramError> {
        let account_info_iter = &mut accounts.iter();

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let amm_lp_mint_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;

        let user_source_info = next_account_info(account_info_iter)?;
        let user_destination_info = next_account_info(account_info_iter)?;
        let user_source_owner = next_account_info(account_info_iter)?;

        Self::check_account_readonly(amm_info)?;
        Self::check_account_readonly(amm_open_orders_info)?;
        Self::check_account_readonly(amm_target_orders_info)?;
        Self::check_account_readonly(amm_coin_vault_info)?;
        Self::check_account_readonly(amm_pc_vault_info)?;
        Self::check_account_readonly(amm_lp_mint_info)?;
        Self::check_account_readonly(market_info)?;
        Self::check_account_readonly(market_event_queue_info)?;
        Self::check_account_readonly(user_source_info)?;
        Self::check_account_readonly(user_destination_info)?;
        Self::check_account_readonly(user_source_owner)?;

        let mut swap_base_in: GetSwapBaseInData = Default::default();
        if let Some(swap) = simulate.swap_base_in_value {
            swap_base_in.amount_in = swap.amount_in;

            if !user_source_owner.is_signer {
                return Err(AmmError::InvalidSignAccount.into());
            }
            let amm = AmmInfo::load_checked(&amm_info, program_id)?;

            if !AmmStatus::from_u64(amm.status).swap_permission() {
                msg!("simulate_swap_base_in: status {}", amm.status);
                return Err(AmmError::InvalidStatus.into());
            }
            let authority = Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?;
            check_assert_eq!(
                *amm_authority_info.key,
                authority,
                "authority",
                AmmError::InvalidProgramAddress
            );
            check_assert_eq!(
                *amm_coin_vault_info.key,
                amm.coin_vault,
                "coin_vault",
                AmmError::InvalidCoinVault
            );
            check_assert_eq!(
                *amm_pc_vault_info.key,
                amm.pc_vault,
                "pc_vault",
                AmmError::InvalidPCVault
            );
            check_assert_eq!(
                *amm_open_orders_info.key,
                amm.open_orders,
                "open_orders",
                AmmError::InvalidOpenOrders
            );
            check_assert_eq!(
                *amm_target_orders_info.key,
                amm.target_orders,
                "target_orders",
                AmmError::InvalidTargetOrders
            );
            check_assert_eq!(
                *amm_lp_mint_info.key,
                amm.lp_mint,
                "lp_mint",
                AmmError::InvalidPoolMint
            );
            check_assert_eq!(
                *market_program_info.key,
                amm.market_program,
                "market_program",
                AmmError::InvalidMarketProgram
            );
            check_assert_eq!(
                *market_info.key,
                amm.market,
                "market",
                AmmError::InvalidMarket
            );

            if *user_source_info.key == amm.pc_vault || *user_source_info.key == amm.coin_vault {
                return Err(AmmError::InvalidInput.into());
            }
            if *user_destination_info.key == amm.pc_vault
                || *user_destination_info.key == amm.coin_vault
            {
                return Err(AmmError::InvalidInput.into());
            }

            let (market_state, open_orders) = Self::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;

            let amm_coin_vault =
                Self::unpack_token_account(&amm_coin_vault_info, &spl_token::id())?;
            let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, &spl_token::id())?;
            let lp_mint = Self::unpack_mint(&amm_lp_mint_info, &spl_token::id())?;

            let user_source = Self::unpack_token_account(&user_source_info, &spl_token::id())?;
            let user_destination =
                Self::unpack_token_account(&user_destination_info, &spl_token::id())?;

            // let target_orders = TargetOrders::load_mut_checked(&target_orders_info, program_id, amm_info.key)?;

            let swap_direction;
            if user_source.mint == amm_coin_vault.mint && user_destination.mint == amm_pc_vault.mint
            {
                swap_direction = SwapDirection::Coin2PC
            } else if user_source.mint == amm_pc_vault.mint
                && user_destination.mint == amm_coin_vault.mint
            {
                swap_direction = SwapDirection::PC2Coin
            } else {
                return Err(AmmError::InvalidInput.into());
            }
            let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;
            swap_base_in.pool_data.status = amm.status;
            swap_base_in.pool_data.coin_decimals = amm.coin_decimals;
            swap_base_in.pool_data.pc_decimals = amm.pc_decimals;
            swap_base_in.pool_data.lp_decimals = lp_mint.decimals.into();
            swap_base_in.pool_data.pool_lp_supply = amm.lp_amount;
            swap_base_in.pool_data.pool_open_time = amm.state_data.pool_open_time;
            swap_base_in.pool_data.pool_pc_amount = total_pc_without_take_pnl;
            swap_base_in.pool_data.pool_coin_amount = total_coin_without_take_pnl;
            swap_base_in.pool_data.amm_id = amm_info.key.to_string();

            let swap_fee = U128::from(swap.amount_in)
                .checked_mul(amm.fees.swap_fee_numerator.into())
                .unwrap()
                .checked_ceil_div(amm.fees.swap_fee_denominator.into())
                .unwrap()
                .0;
            let swap_in_after_deduct_fee =
                U128::from(swap.amount_in).checked_sub(swap_fee).unwrap();
            let swap_amount_out = Calculator::swap_token_amount_base_in(
                swap_in_after_deduct_fee,
                total_pc_without_take_pnl.into(),
                total_coin_without_take_pnl.into(),
                swap_direction,
            )
            .as_u64();
            swap_base_in.minimum_amount_out = swap_amount_out;
            match swap_direction {
                SwapDirection::Coin2PC => {
                    // coin -> pc, need cancel buy order
                    let token_pc_after_swap = total_pc_without_take_pnl
                        .checked_sub(swap_amount_out)
                        .unwrap();
                    let token_coin_after_swap = total_coin_without_take_pnl
                        .checked_add(swap.amount_in)
                        .unwrap();

                    let swap_price_before = total_pc_without_take_pnl
                        .checked_div(total_coin_without_take_pnl)
                        .unwrap();
                    let swap_price_after = token_pc_after_swap
                        .checked_div(token_coin_after_swap)
                        .unwrap();
                    swap_base_in.price_impact =
                        (swap_price_before.checked_sub(swap_price_after).unwrap())
                            .checked_mul(1000000)
                            .unwrap()
                            .checked_div(swap_price_before)
                            .unwrap();
                }
                SwapDirection::PC2Coin => {
                    // pc -> coin, need cancel sell order
                    let token_pc_after_swap = total_pc_without_take_pnl
                        .checked_add(swap.amount_in)
                        .unwrap();
                    let token_coin_after_swap = total_coin_without_take_pnl
                        .checked_sub(swap_amount_out)
                        .unwrap();

                    let swap_price_before = total_pc_without_take_pnl
                        .checked_div(total_coin_without_take_pnl)
                        .unwrap();
                    let swap_price_after = token_pc_after_swap
                        .checked_div(token_coin_after_swap)
                        .unwrap();
                    swap_base_in.price_impact =
                        (swap_price_after.checked_sub(swap_price_before).unwrap())
                            .checked_mul(1000000)
                            .unwrap()
                            .checked_div(swap_price_before)
                            .unwrap();
                }
            }
        }
        return Ok(swap_base_in);
    }

    fn simulate_swap_base_out(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        simulate: SimulateInstruction,
    ) -> Result<GetSwapBaseOutData, ProgramError> {
        let account_info_iter = &mut accounts.iter();

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let amm_lp_mint_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;

        let user_source_info = next_account_info(account_info_iter)?;
        let user_destination_info = next_account_info(account_info_iter)?;
        let user_source_owner = next_account_info(account_info_iter)?;

        Self::check_account_readonly(amm_info)?;
        Self::check_account_readonly(amm_open_orders_info)?;
        Self::check_account_readonly(amm_target_orders_info)?;
        Self::check_account_readonly(amm_coin_vault_info)?;
        Self::check_account_readonly(amm_pc_vault_info)?;
        Self::check_account_readonly(amm_lp_mint_info)?;
        Self::check_account_readonly(market_info)?;
        Self::check_account_readonly(market_event_queue_info)?;
        Self::check_account_readonly(user_source_info)?;
        Self::check_account_readonly(user_destination_info)?;
        Self::check_account_readonly(user_source_owner)?;

        let mut swap_base_out: GetSwapBaseOutData = Default::default();
        if let Some(swap) = simulate.swap_base_out_value {
            swap_base_out.amount_out = swap.amount_out;

            if !user_source_owner.is_signer {
                return Err(AmmError::InvalidSignAccount.into());
            }
            let amm = AmmInfo::load_checked(&amm_info, program_id)?;
            if !AmmStatus::from_u64(amm.status).swap_permission() {
                msg!("simulate_swap_base_out: status {}", amm.status);
                return Err(AmmError::InvalidStatus.into());
            }
            let authority = Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?;
            check_assert_eq!(
                *amm_authority_info.key,
                authority,
                "authority",
                AmmError::InvalidProgramAddress
            );
            check_assert_eq!(
                *amm_coin_vault_info.key,
                amm.coin_vault,
                "coin_vault",
                AmmError::InvalidCoinVault
            );
            check_assert_eq!(
                *amm_pc_vault_info.key,
                amm.pc_vault,
                "pc_vault",
                AmmError::InvalidPCVault
            );
            check_assert_eq!(
                *amm_open_orders_info.key,
                amm.open_orders,
                "open_orders",
                AmmError::InvalidOpenOrders
            );
            check_assert_eq!(
                *amm_target_orders_info.key,
                amm.target_orders,
                "target_orders",
                AmmError::InvalidTargetOrders
            );
            check_assert_eq!(
                *amm_lp_mint_info.key,
                amm.lp_mint,
                "lp_mint",
                AmmError::InvalidPoolMint
            );
            check_assert_eq!(
                *market_program_info.key,
                amm.market_program,
                "market_program",
                AmmError::InvalidMarketProgram
            );
            check_assert_eq!(
                *market_info.key,
                amm.market,
                "market",
                AmmError::InvalidMarket
            );

            if *user_source_info.key == amm.pc_vault || *user_source_info.key == amm.coin_vault {
                return Err(AmmError::InvalidInput.into());
            }
            if *user_destination_info.key == amm.pc_vault
                || *user_destination_info.key == amm.coin_vault
            {
                return Err(AmmError::InvalidInput.into());
            }

            let (market_state, open_orders) = Self::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;

            let amm_coin_vault =
                Self::unpack_token_account(&amm_coin_vault_info, &spl_token::id())?;
            let amm_pc_vault = Self::unpack_token_account(&amm_pc_vault_info, &spl_token::id())?;
            let lp_mint = Self::unpack_mint(&amm_lp_mint_info, &spl_token::id())?;

            let user_swap_source = Self::unpack_token_account(&user_source_info, &spl_token::id())?;
            let user_swap_destination =
                Self::unpack_token_account(&user_destination_info, &spl_token::id())?;

            // let target_orders = TargetOrders::load_mut_checked(&target_orders_info, program_id, amm_info.key)?;

            let swap_direction;
            if user_swap_source.mint == amm_coin_vault.mint
                && user_swap_destination.mint == amm_pc_vault.mint
            {
                swap_direction = SwapDirection::Coin2PC
            } else if user_swap_source.mint == amm_pc_vault.mint
                && user_swap_destination.mint == amm_coin_vault.mint
            {
                swap_direction = SwapDirection::PC2Coin
            } else {
                return Err(AmmError::InvalidInput.into());
            }
            let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;
            swap_base_out.pool_data.status = amm.status;
            swap_base_out.pool_data.coin_decimals = amm.coin_decimals;
            swap_base_out.pool_data.pc_decimals = amm.pc_decimals;
            swap_base_out.pool_data.lp_decimals = lp_mint.decimals.into();
            swap_base_out.pool_data.pool_lp_supply = amm.lp_amount;
            swap_base_out.pool_data.pool_open_time = amm.state_data.pool_open_time;
            swap_base_out.pool_data.pool_pc_amount = total_pc_without_take_pnl;
            swap_base_out.pool_data.pool_coin_amount = total_coin_without_take_pnl;
            swap_base_out.pool_data.amm_id = amm_info.key.to_string();

            let swap_in_before_add_fee = Calculator::swap_token_amount_base_out(
                swap.amount_out.into(),
                total_pc_without_take_pnl.into(),
                total_coin_without_take_pnl.into(),
                swap_direction,
            );

            // swap_in_after_add_fee * (1 - 0.0025) = swap_in_before_add_fee
            // swap_in_after_add_fee = swap_in_before_add_fee / (1 - 0.0025)
            let swap_in_after_add_fee = swap_in_before_add_fee
                .checked_mul(amm.fees.swap_fee_denominator.into())
                .unwrap()
                .checked_ceil_div(
                    (amm.fees
                        .swap_fee_denominator
                        .checked_sub(amm.fees.swap_fee_numerator)
                        .unwrap())
                    .into(),
                )
                .unwrap()
                .0
                .as_u64();
            swap_base_out.max_amount_in = swap_in_after_add_fee;

            match swap_direction {
                SwapDirection::Coin2PC => {
                    // coin -> pc, need cancel buy order
                    let token_pc_after_swap = total_pc_without_take_pnl
                        .checked_sub(swap.amount_out)
                        .unwrap();
                    let token_coin_after_swap = total_coin_without_take_pnl
                        .checked_add(swap_in_after_add_fee)
                        .unwrap();

                    let swap_price_before = total_pc_without_take_pnl
                        .checked_div(total_coin_without_take_pnl)
                        .unwrap();
                    let swap_price_after = token_pc_after_swap
                        .checked_div(token_coin_after_swap)
                        .unwrap();
                    swap_base_out.price_impact =
                        (swap_price_before.checked_sub(swap_price_after).unwrap())
                            .checked_mul(1000000)
                            .unwrap()
                            .checked_div(swap_price_before)
                            .unwrap();
                }
                SwapDirection::PC2Coin => {
                    // pc -> coin, need cancel sell order
                    let token_pc_after_swap = total_pc_without_take_pnl
                        .checked_add(swap_in_after_add_fee)
                        .unwrap();
                    let token_coin_after_swap = total_coin_without_take_pnl
                        .checked_sub(swap.amount_out)
                        .unwrap();

                    let swap_price_before = total_pc_without_take_pnl
                        .checked_div(total_coin_without_take_pnl)
                        .unwrap();
                    let swap_price_after = token_pc_after_swap
                        .checked_div(token_coin_after_swap)
                        .unwrap();
                    swap_base_out.price_impact =
                        (swap_price_after.checked_sub(swap_price_before).unwrap())
                            .checked_mul(1000000)
                            .unwrap()
                            .checked_div(swap_price_before)
                            .unwrap();
                }
            }
        }
        return Ok(swap_base_out);
    }

    fn simulate_run_crank(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
    ) -> Result<RunCrankData, ProgramError> {
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;
        let clock_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;

        Self::check_account_readonly(amm_info)?;
        Self::check_account_readonly(amm_open_orders_info)?;
        Self::check_account_readonly(amm_target_orders_info)?;
        Self::check_account_readonly(amm_coin_vault_info)?;
        Self::check_account_readonly(amm_pc_vault_info)?;
        Self::check_account_readonly(market_info)?;
        Self::check_account_readonly(market_bids_info)?;
        Self::check_account_readonly(market_asks_info)?;
        Self::check_account_readonly(market_event_queue_info)?;

        let amm = AmmInfo::load_checked(amm_info, program_id)?;
        Self::check_accounts(
            program_id,
            &amm,
            amm_info,
            token_program_info,
            clock_info,
            market_program_info,
            amm_authority_info,
            market_info,
            amm_open_orders_info,
            amm_coin_vault_info,
            amm_pc_vault_info,
            amm_target_orders_info,
            None,
            None,
        )?;
        let spl_token_program_id = token_program_info.key;
        let amm_status = AmmStatus::from_u64(amm.status);
        let amm_state = AmmState::from_u64(amm.state);
        let mut run_crank_data = RunCrankData {
            status: amm.status,
            state: amm.state,
            run_crank: false,
        };

        if amm.reset_flag == AmmResetFlag::ResetYes.into_u64() {
            run_crank_data.run_crank = true;
        } else if amm.order_num == 0 {
            run_crank_data.run_crank = false;
        } else {
            match amm_status {
                AmmStatus::Uninitialized
                | AmmStatus::Disabled
                | AmmStatus::WithdrawOnly
                | AmmStatus::LiquidityOnly
                | AmmStatus::SwapOnly
                | AmmStatus::WaitingTrade => {
                    run_crank_data.run_crank = false;
                }
                AmmStatus::Initialized | AmmStatus::OrderBookOnly => match amm_state {
                    AmmState::IdleState => {
                        let amm_coin_vault = Processor::unpack_token_account(
                            &amm_coin_vault_info,
                            spl_token_program_id,
                        )?;
                        let amm_pc_vault = Processor::unpack_token_account(
                            &amm_pc_vault_info,
                            spl_token_program_id,
                        )?;
                        let target = TargetOrders::load_checked(
                            &amm_target_orders_info,
                            program_id,
                            amm_info.key,
                        )?;
                        let (market_state, open_orders) = Processor::load_serum_market_order(
                            market_info,
                            amm_open_orders_info,
                            amm_authority_info,
                            &amm,
                            false,
                        )?;
                        let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
                        let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
                        let (bids, asks) =
                            Self::get_amm_orders(&open_orders, bids_orders, asks_orders)?;
                        let native_pc_total = open_orders.native_pc_total;
                        let native_coin_total = open_orders.native_coin_total;
                        msg!(&format!("simulate_run_crank pc_amount:{}, native_pc_total:{}, coin_amount:{}, native_coin_total:{}, need_take_pnl_pc:{}, need_take_pnl_coin:{}", amm_pc_vault.amount, native_pc_total, amm_coin_vault.amount, native_coin_total, amm.state_data.need_take_pnl_pc, amm.state_data.need_take_pnl_coin));
                        let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                            Calculator::calc_total_without_take_pnl(
                                amm_pc_vault.amount,
                                amm_coin_vault.amount,
                                &open_orders,
                                &amm,
                                &market_state,
                                &market_event_queue_info,
                                &amm_open_orders_info,
                            )?;
                        let x = Calculator::normalize_decimal_v2(
                            total_pc_without_take_pnl,
                            amm.pc_decimals,
                            amm.sys_decimal_value,
                        );
                        let y = Calculator::normalize_decimal_v2(
                            total_coin_without_take_pnl,
                            amm.coin_decimals,
                            amm.sys_decimal_value,
                        );
                        msg!(&format!(
                            "simulate_run_crank x:{}, y:{}, place_x:{}, place_y:{}",
                            x, y, target.placed_x, target.placed_y
                        ));
                        if x.is_zero() || y.is_zero() {
                            run_crank_data.run_crank = false;
                        }

                        let valid_buy_order_num = target.valid_buy_order_num as usize;
                        let valid_sell_order_num = target.valid_sell_order_num as usize;
                        msg!(&format!("simulate_run_crank buy_len:{}, sell_len:{}, plan_buy_order:{}, plan_sell_order:{}", bids.len(), asks.len(), valid_buy_order_num, valid_sell_order_num));
                        if bids.len() < valid_buy_order_num
                            || asks.len() < valid_sell_order_num
                            || (bids.is_empty() && asks.is_empty())
                            || x != target.placed_x.into()
                            || y != target.placed_y.into()
                        {
                            run_crank_data.run_crank = true;
                        }

                        if (x.checked_mul(y).unwrap()
                            < U128::from(target.calc_pnl_x)
                                .checked_mul(target.calc_pnl_y.into())
                                .unwrap())
                            && bids.is_empty()
                            && asks.is_empty()
                        {
                            msg!(arrform!(
                                LOG_SIZE,
                                "{}, {}, {}, {}",
                                x,
                                y,
                                target.calc_pnl_x,
                                target.calc_pnl_y
                            )
                            .as_str());
                            run_crank_data.run_crank = false;
                        }

                        let cur_price: u64 = u64::try_from(
                            (x).checked_mul(amm.sys_decimal_value.into())
                                .unwrap()
                                .checked_div(y)
                                .unwrap()
                                .as_u128(),
                        )
                        .unwrap();
                        let mux_cur_price = (amm.pc_lot_size as u128)
                            .checked_mul(amm.max_price_multiplier as u128)
                            .unwrap();
                        let min_cur_price = (amm.pc_lot_size as u128)
                            .checked_mul(amm.min_price_multiplier as u128)
                            .unwrap();
                        if (cur_price as u128) < min_cur_price
                            || (cur_price as u128) > mux_cur_price
                        {
                            msg!(&format!("simulate_run_crank cur_price:{}, min_cur_price:{}, mux_cur_price:{}", cur_price, min_cur_price, mux_cur_price));
                            run_crank_data.run_crank = false;
                        }
                    }
                    AmmState::CancelAllOrdersState => {
                        let amm_coin_vault = Processor::unpack_token_account(
                            &amm_coin_vault_info,
                            spl_token_program_id,
                        )?;
                        let amm_pc_vault = Processor::unpack_token_account(
                            &amm_pc_vault_info,
                            spl_token_program_id,
                        )?;
                        let target = TargetOrders::load_checked(
                            &amm_target_orders_info,
                            program_id,
                            amm_info.key,
                        )?;
                        let (market_state, open_orders) = Processor::load_serum_market_order(
                            market_info,
                            amm_open_orders_info,
                            amm_authority_info,
                            &amm,
                            false,
                        )?;
                        let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                            Calculator::calc_total_without_take_pnl(
                                amm_pc_vault.amount,
                                amm_coin_vault.amount,
                                &open_orders,
                                &amm,
                                &market_state,
                                &market_event_queue_info,
                                &amm_open_orders_info,
                            )?;
                        let x = Calculator::normalize_decimal_v2(
                            total_pc_without_take_pnl,
                            amm.pc_decimals,
                            amm.sys_decimal_value,
                        );
                        let y = Calculator::normalize_decimal_v2(
                            total_coin_without_take_pnl,
                            amm.coin_decimals,
                            amm.sys_decimal_value,
                        );
                        let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
                        let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
                        let (bids, asks) =
                            Self::get_amm_orders(&open_orders, bids_orders, asks_orders)?;
                        if open_orders.free_slot_bits.count_zeros() > 100
                            && bids.is_empty()
                            && asks.is_empty()
                        {
                            run_crank_data.run_crank = false;
                        } else {
                            run_crank_data.run_crank = true;
                        }
                        if (x.checked_mul(y).unwrap()
                            < U128::from(target.calc_pnl_x)
                                .checked_mul(target.calc_pnl_y.into())
                                .unwrap())
                            && bids.is_empty()
                            && asks.is_empty()
                        {
                            msg!(arrform!(
                                LOG_SIZE,
                                "{}, {}, {}, {}",
                                x,
                                y,
                                target.calc_pnl_x,
                                target.calc_pnl_y
                            )
                            .as_str());
                            run_crank_data.run_crank = false;
                        }
                    }
                    AmmState::PlanOrdersState
                    | AmmState::CancelOrderState
                    | AmmState::PlaceOrdersState
                    | AmmState::PurgeOrderState => {
                        run_crank_data.run_crank = true;
                    }
                    _ => {
                        run_crank_data.run_crank = false;
                    }
                },
            }
        }

        return Ok(run_crank_data);
    }

    /// simulate_info
    pub fn process_simulate_info(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        simulate: SimulateInstruction,
    ) -> ProgramResult {
        let param = simulate.param;
        match SimulateParams::from_u64(param as u64) {
            SimulateParams::PoolInfo => {
                let pool_info_data = Self::simulate_pool_info(program_id, accounts).unwrap();
                msg!("GetPoolData: {}", pool_info_data.to_json());
            }
            SimulateParams::RunCrankInfo => {
                let run_crank_data = Self::simulate_run_crank(program_id, accounts).unwrap();
                msg!("RunCrankData: {}", run_crank_data.to_json());
            }
            SimulateParams::SwapBaseInInfo => {
                let swap_base_in_data =
                    Self::simulate_swap_base_in(program_id, accounts, simulate).unwrap();
                msg!("GetSwapBaseInData: {}", swap_base_in_data.to_json());
            }
            SimulateParams::SwapBaseOutInfo => {
                let swap_base_out_data =
                    Self::simulate_swap_base_out(program_id, accounts, simulate).unwrap();
                msg!("GetSwapBaseOutData: {}", swap_base_out_data.to_json());
            }
        }
        return Ok(());
    }

    pub fn do_idle_state(args: account_parser::IdleArgs) -> ProgramResult {
        let account_parser::IdleArgs {
            program_id: _,
            total_coin_without_take_pnl,
            total_pc_without_take_pnl,
            amm,
            bids,
            asks,
            target,
        } = args;

        let x = Calculator::normalize_decimal_v2(
            *total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y = Calculator::normalize_decimal_v2(
            *total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        if x.is_zero() || y.is_zero() {
            return Err(AmmError::CheckedEmptyFunds.into());
        }

        let valid_buy_order_num = target.valid_buy_order_num as usize;
        let valid_sell_order_num = target.valid_sell_order_num as usize;
        if bids.len() < valid_buy_order_num
            || asks.len() < valid_sell_order_num
            || (bids.is_empty() && asks.is_empty())
            || x != U128::from(target.placed_x)
            || y != U128::from(target.placed_y)
        {
            // clear plan pos before plan
            target.plan_orders_cur = 0;
            amm.state = AmmState::PlanOrdersState.into_u64();
        } else {
            // no order traded
            amm.state = AmmState::IdleState.into_u64();
        }

        let cur_price = U128::from(x)
            .checked_mul(amm.sys_decimal_value.into())
            .unwrap()
            .checked_div(y.into())
            .unwrap();
        let mux_cur_price = U128::from(amm.pc_lot_size)
            .checked_mul(amm.max_price_multiplier.into())
            .unwrap();
        let min_cur_price = U128::from(amm.pc_lot_size)
            .checked_mul(amm.min_price_multiplier.into())
            .unwrap();
        if cur_price < min_cur_price || cur_price > mux_cur_price {
            msg!(arrform!(
                LOG_SIZE,
                "do_idle cur_price:{}, min_cur_price:{}, mux_cur_price:{}",
                cur_price,
                min_cur_price,
                mux_cur_price
            )
            .as_str());
            if bids.is_empty() && asks.is_empty() {
                amm.state = AmmState::IdleState.into_u64();
            } else {
                amm.state = AmmState::CancelAllOrdersState.into_u64();
            }
        }

        if x.checked_mul(y).unwrap()
            < U128::from(target.calc_pnl_x)
                .checked_mul(target.calc_pnl_y.into())
                .unwrap()
        {
            amm.state = AmmState::CancelAllOrdersState.into_u64();
        } else {
            // calc and update pnl
            let (delta_x, delta_y) = Self::calc_take_pnl(
                &target,
                amm,
                total_pc_without_take_pnl,
                total_coin_without_take_pnl,
                x.as_u128().into(),
                y.as_u128().into(),
            )?;
            if delta_x != 0 && delta_y != 0 {
                target.calc_pnl_x = x.checked_sub(U128::from(delta_x)).unwrap().as_u128();
                target.calc_pnl_y = y.checked_sub(U128::from(delta_y)).unwrap().as_u128();
            }
        }

        msg!("do_idle to_state {}", amm.state);
        Ok(())
    }

    /// plan orderbook
    pub fn do_plan_orderbook(args: account_parser::PlanOrderBookArgs) -> ProgramResult {
        let account_parser::PlanOrderBookArgs {
            program_id: _,
            limit,
            total_coin_without_take_pnl,
            total_pc_without_take_pnl,
            amm,
            target,
        } = args;

        if amm.order_num == target.plan_orders_cur {
            // all orders have been planed
            // change to next state
            target.place_orders_cur = 0;
            target.replace_buy_client_id = [0u64; 10];
            target.replace_sell_client_id = [0u64; 10];
            target.placed_x = 0;
            target.placed_y = 0;
            amm.state = AmmState::PlaceOrdersState.into_u64();
        } else {
            let x = Calculator::normalize_decimal_v2(
                total_pc_without_take_pnl,
                amm.pc_decimals,
                amm.sys_decimal_value,
            );
            let y = Calculator::normalize_decimal_v2(
                total_coin_without_take_pnl,
                amm.coin_decimals,
                amm.sys_decimal_value,
            );
            if target.plan_orders_cur == 0 {
                // save tartget_x and target_y
                target.target_x = x.as_u128();
                target.target_y = y.as_u128();
                target.plan_x_buy = x.as_u128();
                target.plan_y_buy = y.as_u128();
                target.plan_x_sell = x.as_u128();
                target.plan_y_sell = y.as_u128();
                target.valid_buy_order_num = 0;
                target.valid_sell_order_num = 0;
            }
            if x.is_zero() || y.is_zero() {
                return Err(AmmError::CheckedEmptyFunds.into());
            }
            if x != target.target_x.into() || y != target.target_y.into() {
                amm.state = AmmState::IdleState.into_u64();
                msg!("do_plan: to_state {}", amm.state);
                return Ok(());
            }
            let min_size: u64 = amm.min_size as u64;
            let cur_price: u64 = u64::try_from(
                (x).checked_mul(amm.sys_decimal_value.into())
                    .unwrap()
                    .checked_div(y)
                    .unwrap()
                    .as_u128(),
            )
            .unwrap();
            let mux_cur_price = (amm.pc_lot_size as u128)
                .checked_mul(amm.max_price_multiplier as u128)
                .unwrap();
            let min_cur_price = (amm.pc_lot_size as u128)
                .checked_mul(amm.min_price_multiplier as u128)
                .unwrap();
            if (cur_price as u128) < min_cur_price || (cur_price as u128) > mux_cur_price {
                msg!(arrform!(
                    LOG_SIZE,
                    "do_plan cur_price:{}, min_cur_price:{}, mux_cur_price:{}",
                    cur_price,
                    min_cur_price,
                    mux_cur_price
                )
                .as_str());
                amm.state = AmmState::IdleState.into_u64();
            } else {
                //let max_bid: u64 = cur_price * (amm.sys_decimal_value - (amm.min_separate + amm.fee)) / amm.sys_decimal_value;
                let max_bid: u64 = U128::from(cur_price)
                    .checked_mul(
                        (amm.fees.trade_fee_denominator
                            - (amm.fees.min_separate_numerator + amm.fees.trade_fee_numerator))
                            .into(),
                    )
                    .unwrap()
                    .checked_div(amm.fees.trade_fee_denominator.into())
                    .unwrap()
                    .as_u64();
                //let min_ask: u64 = cur_price * (amm.sys_decimal_value + amm.min_separate + amm.fee) / amm.sys_decimal_value;
                let min_ask: u64 = U128::from(cur_price)
                    .checked_mul(
                        (amm.fees.trade_fee_denominator
                            + (amm.fees.min_separate_numerator + amm.fees.trade_fee_numerator))
                            .into(),
                    )
                    .unwrap()
                    .checked_ceil_div(amm.fees.trade_fee_denominator.into())
                    .unwrap()
                    .0
                    .as_u64();

                //let grid: u64 = cur_price * (amm.depth as u64) / (100) / (amm.order_num as u64); // percent: e.g., 5*10**6 is 5%
                let mut grid: u64 = cur_price
                    .checked_mul(amm.depth)
                    .unwrap()
                    .checked_div(100)
                    .unwrap()
                    .checked_div(amm.order_num)
                    .unwrap();
                if grid < amm.pc_lot_size {
                    grid = amm.pc_lot_size;
                }
                // msg!("max_bid:{}, min_ask:{}, grid:{}", max_bid, min_ask, grid);
                let plan_orders_num;
                let plan_orders_cur = target.plan_orders_cur;
                if plan_orders_cur + limit as u64 > amm.order_num {
                    plan_orders_num = amm.order_num;
                } else {
                    plan_orders_num = plan_orders_cur + limit as u64;
                }
                let fb = Calculator::fibonacci(amm.order_num);
                // To plan orders and update target
                for i in plan_orders_cur..plan_orders_num {
                    let distance = grid.checked_mul(fb[i as usize]).unwrap();
                    // plan buy side
                    // buy_price = max_bid - grid * i;
                    let mut buy_price = max_bid.saturating_sub(distance);
                    // change the grid distance for last buy order
                    if i == amm.order_num.checked_sub(1).unwrap()
                        && target.last_order_denominator != 0
                        && target.last_order_numerator != 0
                    {
                        let target_buy_price = cur_price
                            .checked_mul(target.last_order_denominator)
                            .unwrap()
                            .checked_div(target.last_order_numerator)
                            .unwrap();
                        if buy_price > target_buy_price {
                            buy_price = target_buy_price;
                        }
                        // change the price to ticksize if the cur price is invalid and the last price is valid
                        if buy_price < amm.pc_lot_size
                            && target.buy_orders[i as usize - 2].price != 0
                            && target.buy_orders[i as usize - 2].price != grid
                        {
                            buy_price = grid;
                        }
                    }

                    let mut buy_vol = 0u64;
                    buy_price = Calculator::floor_lot(buy_price, amm.pc_lot_size as u64);
                    if buy_price != 0u64 {
                        let max_buy_vol = Calculator::get_max_buy_size_at_price(
                            buy_price,
                            target.plan_x_buy,
                            target.plan_y_buy,
                            &amm,
                        );
                        // msg!("buy_price:{}, max_buy_vol:{}", buy_price, max_buy_vol);
                        buy_vol = max_buy_vol
                            .checked_sub(Calculator::to_u64(
                                U128::from(amm.vol_max_cut_ratio)
                                    .checked_mul(max_buy_vol.into())
                                    .unwrap()
                                    .checked_div(U128::from(TEN_THOUSAND))
                                    .unwrap()
                                    .as_u128(),
                            )?)
                            .unwrap();
                    }
                    if buy_vol < min_size {
                        buy_vol = 0u64;
                    }
                    let max_buy_qty_u64 = Calculator::floor_lot(
                        buy_vol,
                        Calculator::normalize_decimal(
                            amm.coin_lot_size as u64,
                            amm.coin_decimals,
                            amm.sys_decimal_value,
                        ),
                    );
                    target.plan_x_buy = target
                        .plan_x_buy
                        .checked_sub(
                            U128::from(max_buy_qty_u64)
                                .checked_mul(buy_price.into())
                                .unwrap()
                                .checked_div(amm.sys_decimal_value.into())
                                .unwrap()
                                .as_u128(),
                        )
                        .unwrap();
                    target.plan_y_buy = target
                        .plan_y_buy
                        .checked_add(max_buy_qty_u64.into())
                        .unwrap();
                    // update tartget buy orders
                    target.buy_orders[i as usize].price = buy_price;
                    target.buy_orders[i as usize].vol = max_buy_qty_u64;

                    // plan sell side
                    // sell_price = min_ask + grid * i
                    let mut sell_price = min_ask.checked_add(distance).unwrap();
                    // change the last sell order price
                    if i == amm.order_num.checked_sub(1).unwrap()
                        && target.last_order_denominator != 0
                        && target.last_order_numerator != 0
                    {
                        let target_sell_price = cur_price
                            .checked_mul(target.last_order_numerator)
                            .unwrap()
                            .checked_div(target.last_order_denominator)
                            .unwrap();
                        if sell_price < target_sell_price {
                            sell_price = target_sell_price;
                        }
                    }
                    sell_price = Calculator::ceil_lot(sell_price, amm.pc_lot_size);
                    let max_sell_vol = Calculator::get_max_sell_size_at_price(
                        sell_price,
                        target.plan_x_sell,
                        target.plan_y_sell,
                        &amm,
                    );
                    // msg!("sell_price:{}, max_sell_vol:{}", sell_price, max_sell_vol);
                    let mut sell_vol = max_sell_vol
                        .checked_sub(Calculator::to_u64(
                            U128::from(amm.vol_max_cut_ratio)
                                .checked_mul(max_sell_vol.into())
                                .unwrap()
                                .checked_div(U128::from(TEN_THOUSAND))
                                .unwrap()
                                .as_u128(),
                        )?)
                        .unwrap();
                    if sell_vol < min_size {
                        sell_vol = 0u64;
                    }
                    let max_sell_qty_u64 = Calculator::floor_lot(
                        sell_vol,
                        Calculator::normalize_decimal(
                            amm.coin_lot_size as u64,
                            amm.coin_decimals,
                            amm.sys_decimal_value,
                        ),
                    );
                    target.plan_y_sell = target
                        .plan_y_sell
                        .checked_sub(max_sell_qty_u64.into())
                        .unwrap();
                    target.plan_x_sell = target
                        .plan_x_sell
                        .checked_add(
                            U128::from(max_sell_qty_u64)
                                .checked_mul(sell_price.into())
                                .unwrap()
                                .checked_div(amm.sys_decimal_value.into())
                                .unwrap()
                                .as_u128(),
                        )
                        .unwrap();
                    target.sell_orders[i as usize].price = sell_price;
                    target.sell_orders[i as usize].vol = max_sell_qty_u64;
                    // msg!(arrform!(LOG_SIZE, "do_plan i:{}, {}, {}, {}, {}", i, target.buy_orders[i as usize].price, target.buy_orders[i as usize].vol, target.sell_orders[i as usize].price, target.sell_orders[i as usize].vol).as_str());

                    // update plan_orders_cur
                    target.plan_orders_cur += 1;
                }
                if amm.order_num == target.plan_orders_cur {
                    // all orders have been planed
                    // change to next state
                    target.place_orders_cur = 0;
                    target.replace_buy_client_id = [0u64; 10];
                    target.replace_sell_client_id = [0u64; 10];
                    target.placed_x = 0;
                    target.placed_y = 0;
                    amm.state = AmmState::PlaceOrdersState.into_u64();
                }
            }
        }
        Ok(())
    }

    pub fn do_cancel_order(args: account_parser::CancelOrderArgs) -> ProgramResult {
        let account_parser::CancelOrderArgs {
            program_id: _,
            limit: _,
            market_program_info: _,
            market_info: _,
            amm_open_orders_info: _,
            amm_authority_info: _,
            market_event_queue_info: _,
            market_bids_info: _,
            market_asks_info: _,
            amm,
            open_orders: _,
            target: _,
            bids: _,
            asks: _,

            coin_amount: _,
            pc_amount: _,
        } = args;

        amm.state = AmmState::IdleState.into_u64();
        Ok(())
    }

    pub fn do_place_orders(args: account_parser::PlaceOrdersArgs) -> ProgramResult {
        let account_parser::PlaceOrdersArgs {
            program_id: _,
            limit,

            amm_authority_info,
            amm_open_orders_info,
            market_program_info,
            market_info,
            market_request_queue_info,
            amm_coin_vault_info,
            amm_pc_vault_info,
            market_coin_vault_info,
            market_pc_vault_info,
            token_program_info,
            rent_info,

            market_event_queue_info,
            market_bids_info,
            market_asks_info,

            srm_token_account,

            amm,
            open_orders,
            bids,
            asks,
            target,

            total_coin_without_take_pnl,
            total_pc_without_take_pnl,

            coin_vault_amount,
            pc_vault_amount,
        } = args;

        if open_orders.free_slot_bits.count_zeros() > 100 {
            // TooManyOpenOrders
            amm.state = AmmState::CancelAllOrdersState.into_u64();
            return Ok(());
        }
        let x = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        if x.is_zero() || y.is_zero() {
            return Err(AmmError::CheckedEmptyFunds.into());
        }
        if x != target.target_x.into() || y != target.target_y.into() {
            amm.state = AmmState::IdleState.into_u64();
            return Ok(());
        }
        let place_orders_num;
        let place_orders_cur = target.place_orders_cur;
        let target_orders_buy = target.buy_orders;
        let target_orders_sell = target.sell_orders;
        if place_orders_cur + limit as u64 > amm.order_num {
            place_orders_num = amm.order_num
        } else {
            place_orders_num = place_orders_cur + limit as u64;
        }
        // msg!(arrform!(LOG_SIZE, "do_place num:{}, cur:{}, limit:{}, buy_len:{}, sell_len:{}", place_orders_num, place_orders_cur, limit, bids.len(), asks.len()).as_str());
        let mut bids_client_ids: VecDeque<u64> = bids
            .into_iter()
            .map(|item| item.client_order_id())
            .collect();
        let mut asks_client_ids: VecDeque<u64> = asks
            .into_iter()
            .map(|item| item.client_order_id())
            .collect();
        bids_client_ids.retain(|x| !target.replace_buy_client_id.contains(&x));
        asks_client_ids.retain(|x| !target.replace_sell_client_id.contains(&x));

        let mut pc_avaliable = pc_vault_amount
            .checked_add(open_orders.native_pc_free)
            .unwrap()
            .checked_sub(amm.state_data.need_take_pnl_pc)
            .unwrap();
        let mut coin_avaliable = coin_vault_amount
            .checked_add(open_orders.native_coin_free)
            .unwrap()
            .checked_sub(amm.state_data.need_take_pnl_coin)
            .unwrap();
        for i in place_orders_cur..place_orders_num {
            let payer = amm_pc_vault_info.clone();
            let side = Side::Bid;
            let limit_price_u64 =
                Calculator::convert_price_out(target_orders_buy[i as usize].price, amm.pc_lot_size);
            let max_coin_qty = Calculator::convert_vol_out(
                target_orders_buy[i as usize].vol,
                amm.coin_decimals,
                amm.coin_lot_size,
                amm.sys_decimal_value,
            );
            let order_type = OrderType::Limit;

            let out_pc_lot_size = Calculator::convert_out_pc_lot_size(
                amm.pc_decimals as u8,
                amm.coin_decimals as u8,
                amm.pc_lot_size,
                amm.coin_lot_size,
                amm.sys_decimal_value,
            );
            let max_native_pc_qty_including_fees = max_coin_qty
                .checked_mul(limit_price_u64)
                .unwrap()
                .checked_mul(out_pc_lot_size)
                .unwrap();
            if max_native_pc_qty_including_fees > pc_avaliable {
                // pc amount is InsufficientFunds
                amm.state = AmmState::CancelAllOrdersState.into_u64();
                return Ok(());
            }
            if limit_price_u64 != 0 && max_coin_qty != 0 {
                if bids_client_ids.is_empty() {
                    let client_order_id = amm.incr_client_order_id();
                    // msg!("new buy order:{}, {}, {}", client_order_id, limit_price_u64, max_coin_qty);
                    target.replace_buy_client_id[target.valid_buy_order_num as usize] =
                        client_order_id;
                    Invokers::invoke_dex_new_order_v3(
                        market_program_info.clone(),
                        market_info.clone(),
                        amm_open_orders_info.clone(),
                        market_request_queue_info.clone(),
                        market_event_queue_info.clone(),
                        market_bids_info.clone(),
                        market_asks_info.clone(),
                        payer,
                        amm_authority_info.clone(),
                        market_coin_vault_info.clone(),
                        market_pc_vault_info.clone(),
                        token_program_info.clone(),
                        rent_info.clone(),
                        srm_token_account.clone(),
                        AUTHORITY_AMM,
                        amm.nonce as u8,
                        side,
                        NonZeroU64::new(limit_price_u64).unwrap(),
                        NonZeroU64::new(max_coin_qty).unwrap(),
                        NonZeroU64::new(max_native_pc_qty_including_fees).unwrap(),
                        order_type,
                        client_order_id,
                        std::u16::MAX,
                    )?;
                } else {
                    let client_order_id = if target.valid_buy_order_num == 0 {
                        bids_client_ids.pop_back().unwrap()
                    } else {
                        bids_client_ids.pop_front().unwrap()
                    };
                    target.replace_buy_client_id[target.valid_buy_order_num as usize] =
                        client_order_id;
                    // msg!("replace buy order:{}, {}, {}", client_order_id, limit_price_u64, max_coin_qty);
                    Invokers::invoke_dex_replace_order_by_client_id(
                        market_program_info.clone(),
                        market_info.clone(),
                        amm_open_orders_info.clone(),
                        market_request_queue_info.clone(),
                        market_event_queue_info.clone(),
                        market_bids_info.clone(),
                        market_asks_info.clone(),
                        payer,
                        amm_authority_info.clone(),
                        market_coin_vault_info.clone(),
                        market_pc_vault_info.clone(),
                        token_program_info.clone(),
                        rent_info.clone(),
                        srm_token_account.clone(),
                        AUTHORITY_AMM,
                        amm.nonce as u8,
                        side,
                        NonZeroU64::new(limit_price_u64).unwrap(),
                        NonZeroU64::new(max_coin_qty).unwrap(),
                        NonZeroU64::new(max_native_pc_qty_including_fees).unwrap(),
                        order_type,
                        client_order_id,
                        std::u16::MAX,
                    )?;
                }
                target.valid_buy_order_num += 1;
            }

            let payer = amm_coin_vault_info.clone();
            let side = Side::Ask;
            let limit_price_u64 = Calculator::convert_price_out(
                target_orders_sell[i as usize].price,
                amm.pc_lot_size,
            );
            let max_coin_qty = Calculator::convert_vol_out(
                target_orders_sell[i as usize].vol,
                amm.coin_decimals,
                amm.coin_lot_size,
                amm.sys_decimal_value,
            );
            let order_type = OrderType::Limit;

            if max_coin_qty.checked_mul(amm.coin_lot_size).unwrap() > coin_avaliable {
                // coin amount is InsufficientFunds
                amm.state = AmmState::CancelAllOrdersState.into_u64();
                return Ok(());
            }

            let max_native_pc_qty_including_fees = 1u64;
            if limit_price_u64 != 0 && max_coin_qty != 0 {
                if asks_client_ids.is_empty() {
                    let client_order_id = amm.incr_client_order_id();
                    target.replace_sell_client_id[target.valid_sell_order_num as usize] =
                        client_order_id;
                    // msg!("new sell order:{}, {}, {}", client_order_id, limit_price_u64, max_coin_qty);
                    Invokers::invoke_dex_new_order_v3(
                        market_program_info.clone(),
                        market_info.clone(),
                        amm_open_orders_info.clone(),
                        market_request_queue_info.clone(),
                        market_event_queue_info.clone(),
                        market_bids_info.clone(),
                        market_asks_info.clone(),
                        payer,
                        amm_authority_info.clone(),
                        market_coin_vault_info.clone(),
                        market_pc_vault_info.clone(),
                        token_program_info.clone(),
                        rent_info.clone(),
                        srm_token_account.clone(),
                        AUTHORITY_AMM,
                        amm.nonce as u8,
                        side,
                        NonZeroU64::new(limit_price_u64).unwrap(),
                        NonZeroU64::new(max_coin_qty).unwrap(),
                        NonZeroU64::new(max_native_pc_qty_including_fees).unwrap(),
                        order_type,
                        client_order_id,
                        std::u16::MAX,
                    )?;
                } else {
                    let client_order_id = if target.valid_sell_order_num == 0 {
                        asks_client_ids.pop_back().unwrap()
                    } else {
                        asks_client_ids.pop_front().unwrap()
                    };
                    target.replace_sell_client_id[target.valid_sell_order_num as usize] =
                        client_order_id;
                    // msg!("replace sell order:{}, {}, {}", client_order_id, limit_price_u64, max_coin_qty);
                    Invokers::invoke_dex_replace_order_by_client_id(
                        market_program_info.clone(),
                        market_info.clone(),
                        amm_open_orders_info.clone(),
                        market_request_queue_info.clone(),
                        market_event_queue_info.clone(),
                        market_bids_info.clone(),
                        market_asks_info.clone(),
                        payer,
                        amm_authority_info.clone(),
                        market_coin_vault_info.clone(),
                        market_pc_vault_info.clone(),
                        token_program_info.clone(),
                        rent_info.clone(),
                        srm_token_account.clone(),
                        AUTHORITY_AMM,
                        amm.nonce as u8,
                        side,
                        NonZeroU64::new(limit_price_u64).unwrap(),
                        NonZeroU64::new(max_coin_qty).unwrap(),
                        NonZeroU64::new(max_native_pc_qty_including_fees).unwrap(),
                        order_type,
                        client_order_id,
                        std::u16::MAX,
                    )?;
                }
                target.valid_sell_order_num += 1;
            }
            // update place_orders_cur
            target.place_orders_cur += 1;
            {
                // reload account data to check avaliable pc and avaliable coin is enough to place order
                let open_orders = Self::load_orders(amm_open_orders_info)?;
                let pc_vault = Self::unpack_token_account(&amm_pc_vault_info, &spl_token::id())?;
                let coin_vault = Self::unpack_token_account(&amm_pc_vault_info, &spl_token::id())?;
                pc_avaliable = pc_vault
                    .amount
                    .checked_add(open_orders.native_pc_free)
                    .unwrap()
                    .checked_sub(amm.state_data.need_take_pnl_pc)
                    .unwrap();
                coin_avaliable = coin_vault
                    .amount
                    .checked_add(open_orders.native_coin_free)
                    .unwrap()
                    .checked_sub(amm.state_data.need_take_pnl_coin)
                    .unwrap();
            }
        }
        if target.place_orders_cur == amm.order_num {
            // save placed_x & placed_y
            target.placed_x = x.as_u128();
            target.placed_y = y.as_u128();
        }
        if target.place_orders_cur >= amm.order_num {
            amm.state = AmmState::PurgeOrderState.into_u64();
            return Ok(());
        }
        Ok(())
    }

    pub fn do_purge_orders(args: account_parser::PurgeOrderArgs) -> ProgramResult {
        let account_parser::PurgeOrderArgs {
            program_id: _,
            limit: _,
            market_program_info,
            market_info,
            amm_open_orders_info,
            amm_authority_info,
            market_event_queue_info,
            market_bids_info,
            market_asks_info,
            amm,
            target,
            bids,
            asks,
        } = args;

        if bids.len() <= target.valid_buy_order_num as usize
            && asks.len() <= target.valid_sell_order_num as usize
        {
            // no order to purge
            amm.state = AmmState::IdleState.into_u64();
        } else {
            let mut cancel_client_order_ids = Vec::new();
            for i in target.valid_buy_order_num as usize..bids.len() {
                cancel_client_order_ids.push(bids[i].client_order_id());
            }
            for i in target.valid_sell_order_num as usize..asks.len() {
                cancel_client_order_ids.push(asks[i].client_order_id());
            }

            let mut order_ids_vec = Vec::new();
            let mut order_ids = [0u64; 8];
            let mut count = 0;
            for i in 0..cancel_client_order_ids.len() {
                order_ids[count] = cancel_client_order_ids[i];
                count += 1;
                if count == 8 {
                    order_ids_vec.push(order_ids);
                    order_ids = [0u64; 8];
                    count = 0;
                }
            }
            if count != 0 {
                order_ids_vec.push(order_ids);
            }
            for ids in order_ids_vec.iter() {
                Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                    market_program_info.clone(),
                    market_info.clone(),
                    market_bids_info.clone(),
                    market_asks_info.clone(),
                    amm_open_orders_info.clone(),
                    amm_authority_info.clone(),
                    market_event_queue_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    *ids,
                )?;
            }
        }
        msg!("do_purge to_state {}", amm.state);
        Ok(())
    }

    pub fn do_cancel_all_orders_state(args: account_parser::CancelAllOrdersArgs) -> ProgramResult {
        let account_parser::CancelAllOrdersArgs {
            program_id: _,
            limit: _,
            market_program_info,
            market_info,
            amm_open_orders_info,
            amm_authority_info,
            market_event_queue_info,
            market_coin_vault_info,
            market_pc_vault_info,
            market_bids_info,
            market_asks_info,
            market_vault_signer,
            token_program_info,
            amm_coin_vault_info,
            amm_pc_vault_info,
            referrer_pc_account,
            amm,
            open_orders,
            target,
            bids,
            asks,
        } = args;

        msg!(arrform!(
            LOG_SIZE,
            "do_cancel_all o_bits:{:x}",
            identity(open_orders.free_slot_bits)
        )
        .as_str());
        // to decide whether all orders are really canceled
        if open_orders.free_slot_bits.count_zeros() > 100 && bids.is_empty() && asks.is_empty() {
            msg!(
                "do_cancel_all to state :{}, count_zeros:{}",
                amm.state,
                open_orders.free_slot_bits.count_zeros()
            );
            return Ok(());
        } else {
            let mut amm_order_ids_vec = Vec::new();
            let mut order_ids = [0u64; 8];
            let mut count = 0;
            for i in 0..std::cmp::max(bids.len(), asks.len()) {
                if i < bids.len() {
                    order_ids[count] = bids[i].client_order_id();
                    count += 1;
                }
                if i < asks.len() {
                    order_ids[count] = asks[i].client_order_id();
                    count += 1;
                }
                if count == 8 {
                    amm_order_ids_vec.push(order_ids);
                    order_ids = [0u64; 8];
                    count = 0;
                }
            }
            if count != 0 {
                amm_order_ids_vec.push(order_ids);
            }
            for ids in amm_order_ids_vec.iter() {
                Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                    market_program_info.clone(),
                    market_info.clone(),
                    market_bids_info.clone(),
                    market_asks_info.clone(),
                    amm_open_orders_info.clone(),
                    amm_authority_info.clone(),
                    market_event_queue_info.clone(),
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                    *ids,
                )?;
            }
            if open_orders.native_coin_total != 0 || open_orders.native_pc_total != 0 {
                Invokers::invoke_dex_settle_funds(
                    market_program_info.clone(),
                    market_info.clone(),
                    amm_open_orders_info.clone(),
                    amm_authority_info.clone(),
                    market_coin_vault_info.clone(),
                    market_pc_vault_info.clone(),
                    amm_coin_vault_info.clone(),
                    amm_pc_vault_info.clone(),
                    market_vault_signer.clone(),
                    token_program_info.clone(),
                    referrer_pc_account,
                    AUTHORITY_AMM,
                    amm.nonce as u8,
                )?;
            }
            if amm.reset_flag == AmmResetFlag::ResetYes.into_u64() {
                amm.reset_flag = AmmResetFlag::ResetNo.into_u64();
            }
            amm.state = AmmState::IdleState.into_u64();
            target.plan_orders_cur = 0;
            target.place_orders_cur = 0;
        }
        Ok(())
    }

    pub fn process_set_params(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        setparams: SetParamsInstruction,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer = next_account_info(account_info_iter)?;
        let market_event_q_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;

        let amm_owner_info = next_account_info(account_info_iter)?;

        if *token_program_info.key != spl_token::ID {
            return Err(AmmError::InvalidSplTokenProgram.into());
        }
        let mut amm = AmmInfo::load_mut_checked(&amm_info, program_id)?;
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        if amm_info.owner != program_id {
            return Err(AmmError::InvalidOwner.into());
        }
        if !amm_owner_info.is_signer || *amm_owner_info.key != config_feature::amm_owner::ID {
            return Err(AmmError::InvalidSignAccount.into());
        }
        if *market_program_info.key != amm.market_program {
            return Err(AmmError::InvalidMarketProgram.into());
        }
        if *market_info.key != amm.market {
            return Err(AmmError::InvalidMarket.into());
        }
        if *amm_open_orders_info.key != amm.open_orders {
            return Err(AmmError::InvalidOpenOrders.into());
        }
        if amm.coin_vault != *amm_coin_vault_info.key {
            return Err(AmmError::InvalidCoinVault.into());
        }
        if amm.pc_vault != *amm_pc_vault_info.key {
            return Err(AmmError::InvalidPCVault.into());
        }
        if amm.target_orders != *amm_target_orders_info.key {
            return Err(AmmError::InvalidTargetOrders.into());
        }
        // cancel amm orders in openbook
        Self::do_cancel_amm_orders(
            &amm,
            amm_authority_info,
            amm_open_orders_info,
            market_program_info,
            market_info,
            market_bids_info,
            market_asks_info,
            market_event_q_info,
            AUTHORITY_AMM,
        )?;
        Invokers::invoke_dex_settle_funds(
            market_program_info.clone(),
            market_info.clone(),
            amm_open_orders_info.clone(),
            amm_authority_info.clone(),
            market_coin_vault_info.clone(),
            market_pc_vault_info.clone(),
            amm_coin_vault_info.clone(),
            amm_pc_vault_info.clone(),
            market_vault_signer.clone(),
            token_program_info.clone(),
            None,
            AUTHORITY_AMM,
            amm.nonce as u8,
        )?;

        let param = setparams.param;
        let mut set_valid = false;
        match AmmParams::from_u64(param as u64) {
            AmmParams::Status => {
                {
                    let (market_state, open_orders) = Processor::load_serum_market_order(
                        market_info,
                        amm_open_orders_info,
                        amm_authority_info,
                        &amm,
                        false,
                    )?;
                    let (shared_pc, shared_coin) = Calculator::calc_exact_vault_in_serum(
                        &open_orders,
                        &market_state,
                        market_event_q_info,
                        amm_open_orders_info,
                    )
                    .unwrap();
                    if shared_pc != 0 || shared_coin != 0 {
                        msg!("shared_pc:{}, shared_coin:{}", shared_pc, shared_coin);
                        return Err(AmmError::InvalidInput.into());
                    }
                }
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if AmmStatus::valid_status(value) {
                    amm.status = value as u64;
                    set_valid = true;
                }
            }
            AmmParams::State => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if AmmState::valid_state(value) {
                    amm.state = value as u64;
                    set_valid = true;
                }
            }
            AmmParams::OrderNum => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if value > MAX_ORDER_LIMIT as u64 {
                    return Err(AmmError::InvalidInput.into());
                }
                amm.order_num = value as u64;
                set_valid = true;
            }
            AmmParams::Depth => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if value > 0 && value < 100 {
                    amm.depth = value as u64;
                    set_valid = true;
                }
            }
            AmmParams::AmountWave => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                amm.amount_wave = value;
                set_valid = true;
            }
            AmmParams::MinPriceMultiplier => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if value < amm.max_price_multiplier {
                    amm.min_price_multiplier = value;
                    set_valid = true;
                }
            }
            AmmParams::MaxPriceMultiplier => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if value > amm.max_price_multiplier {
                    amm.max_price_multiplier = value;
                    set_valid = true;
                }
            }
            AmmParams::VolMaxCutRatio => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if value <= TEN_THOUSAND {
                    amm.vol_max_cut_ratio = value;
                    set_valid = true;
                }
            }
            AmmParams::Seperate => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                if value <= TEN_THOUSAND {
                    amm.fees.min_separate_numerator = value;
                    set_valid = true;
                }
            }
            AmmParams::Fees => {
                let fees = match setparams.fees {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                fees.validate()?;
                amm.fees = fees;
                set_valid = true;
            }
            AmmParams::AmmOwner => {
                let new_pubkey = match setparams.new_pubkey {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                amm.amm_owner = new_pubkey;
                set_valid = true;
            }
            AmmParams::SetOpenTime => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                amm.state_data.pool_open_time = value as u64;
                set_valid = true;
            }
            AmmParams::LastOrderDistance => {
                let mut target = TargetOrders::load_mut_checked(
                    &amm_target_orders_info,
                    program_id,
                    amm_info.key,
                )?;
                let distance = match setparams.last_order_distance {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                target.last_order_numerator = distance.last_order_numerator;
                target.last_order_denominator = distance.last_order_denominator;
                set_valid = true;
            }
            AmmParams::InitOrderDepth => {
                amm.order_num = 7u64;
                amm.depth = 3u64;
                set_valid = true;
            }
            AmmParams::SetSwitchTime => {
                let value = match setparams.value {
                    Some(a) => a,
                    None => return Err(AmmError::InvalidInput.into()),
                };
                amm.state_data.orderbook_to_init_time = value as u64;
                set_valid = true;
            }
            AmmParams::ClearOpenTime => {
                amm.state_data.pool_open_time = 0;
                amm.state_data.orderbook_to_init_time = 0;
                set_valid = true;
            }
            AmmParams::UpdateOpenOrder => {
                let new_open_orders_info = next_account_info(account_info_iter)?;
                amm.open_orders = *new_open_orders_info.key;
                let (_market_state, _open_orders) = Processor::load_serum_market_order(
                    market_info,
                    new_open_orders_info,
                    amm_authority_info,
                    &amm,
                    false,
                )?;
                set_valid = true;
            }
            _ => {
                return Err(AmmError::InvalidInput.into());
            }
        }
        if set_valid {
            amm.state = AmmState::CancelAllOrdersState.into_u64();
            amm.reset_flag = AmmResetFlag::ResetYes.into_u64();
        } else {
            return Err(AmmError::InvalidParamsSet.into());
        }
        Ok(())
    }

    pub fn process_monitor_step(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        monitor: MonitorStepInstruction,
    ) -> ProgramResult {
        const MIN_ACCOUNTS: usize = 18;
        let input_account_len = accounts.len();
        if input_account_len != MIN_ACCOUNTS
            && input_account_len != MIN_ACCOUNTS + 1
            && input_account_len != MIN_ACCOUNTS + 2
        {
            return Err(AmmError::WrongAccountsNumber.into());
        }
        let (fixed_accounts, extra_token_accounts) = array_refs![accounts, MIN_ACCOUNTS; .. ;];
        let [token_program_info, rent_info, clock_info, amm_info, amm_authority_info, amm_open_orders_info, amm_target_orders_info, amm_coin_vault_info, amm_pc_vault_info, market_program_info, market_info, market_coin_vault_info, market_pc_vault_info, market_vault_signer, market_request_queue_info, market_event_queue_info, market_bids_info, market_asks_info] =
            fixed_accounts;
        let mut srm_token_info = None;
        let mut referrer_pc_info = None;
        if input_account_len == MIN_ACCOUNTS + 1 {
            if let [srm_token_acc] = extra_token_accounts {
                srm_token_info = Some(srm_token_acc);
            }
        } else if input_account_len == MIN_ACCOUNTS + 2 {
            if let [srm_token_acc, referrer_pc_acc] = extra_token_accounts {
                srm_token_info = Some(srm_token_acc);
                referrer_pc_info = Some(referrer_pc_acc);
            }
        }

        let mut amm = AmmInfo::load_mut_checked(amm_info, program_id)?;
        Self::check_accounts(
            program_id,
            &amm,
            amm_info,
            token_program_info,
            clock_info,
            market_program_info,
            amm_authority_info,
            market_info,
            amm_open_orders_info,
            amm_coin_vault_info,
            amm_pc_vault_info,
            amm_target_orders_info,
            srm_token_info,
            referrer_pc_info,
        )?;
        check_assert_eq!(
            *market_program_info.key,
            config_feature::openbook_program::id(),
            "market_program",
            AmmError::InvalidMarketProgram
        );
        if amm_info.owner != program_id {
            return Err(AmmError::InvalidOwner.into());
        }
        if amm.min_size == 0 {
            return Err(AmmError::MarketLotSizeIsTooLarge.into());
        }
        let amm_status = AmmStatus::from_u64(amm.status);
        let amm_state = AmmState::from_u64(amm.state);
        let spl_token_program_id = token_program_info.key;

        if amm.reset_flag == AmmResetFlag::ResetYes.into_u64() {
            msg!("monitor_step: ResetYes");
            let cancel_all_orders_accounts: &[AccountInfo] = &[
                amm_info.clone(),
                market_program_info.clone(),
                market_info.clone(),
                amm_open_orders_info.clone(),
                amm_authority_info.clone(),
                market_event_queue_info.clone(),
                market_coin_vault_info.clone(),
                market_pc_vault_info.clone(),
                market_bids_info.clone(),
                market_asks_info.clone(),
                market_vault_signer.clone(),
                token_program_info.clone(),
                amm_coin_vault_info.clone(),
                amm_pc_vault_info.clone(),
                amm_target_orders_info.clone(),
            ];
            account_parser::CancelAllOrdersArgs::with_parsed_args(
                program_id,
                monitor.cancel_order_limit,
                &mut amm,
                cancel_all_orders_accounts,
                referrer_pc_info,
                Self::do_cancel_all_orders_state,
            )
            .unwrap();
        } else if amm.order_num == 0 {
            msg!(arrform!(
                LOG_SIZE,
                "monitor_step Status:{}, order_num:{}",
                amm.status,
                amm.order_num
            )
            .as_str());
        } else {
            match amm_status {
                AmmStatus::Uninitialized
                | AmmStatus::Disabled
                | AmmStatus::WithdrawOnly
                | AmmStatus::LiquidityOnly
                | AmmStatus::SwapOnly
                | AmmStatus::WaitingTrade => {
                    msg!("monitor_step: AmmStatus:{}", amm.status);
                    return Err(AmmError::InvalidStatus.into());
                }
                AmmStatus::Initialized | AmmStatus::OrderBookOnly => match amm_state {
                    AmmState::IdleState => {
                        msg!("monitor_step IdleState:{}", amm.state);
                        let idle_accounts: &[AccountInfo] = &[
                            amm_info.clone(),
                            market_program_info.clone(),
                            market_info.clone(),
                            market_bids_info.clone(),
                            market_asks_info.clone(),
                            market_event_queue_info.clone(),
                            amm_authority_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            amm_target_orders_info.clone(),
                        ];
                        account_parser::IdleArgs::with_parsed_args(
                            program_id,
                            spl_token_program_id,
                            &mut amm,
                            idle_accounts,
                            Self::do_idle_state,
                        )
                        .unwrap();
                    }
                    AmmState::CancelAllOrdersState => {
                        msg!("monitor_step CancelAllOrdersState:{}", amm.state);
                        let cancel_all_orders_accounts: &[AccountInfo] = &[
                            amm_info.clone(),
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_event_queue_info.clone(),
                            market_coin_vault_info.clone(),
                            market_pc_vault_info.clone(),
                            market_bids_info.clone(),
                            market_asks_info.clone(),
                            market_vault_signer.clone(),
                            token_program_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            amm_target_orders_info.clone(),
                        ];
                        account_parser::CancelAllOrdersArgs::with_parsed_args(
                            program_id,
                            monitor.cancel_order_limit,
                            &mut amm,
                            cancel_all_orders_accounts,
                            referrer_pc_info,
                            Self::do_cancel_all_orders_state,
                        )
                        .unwrap();
                    }
                    AmmState::PlanOrdersState => {
                        msg!("monitor_step PlanOrdersState:{}", amm.state);
                        let plan_buy_accounts: &[AccountInfo] = &[
                            amm_info.clone(),
                            market_info.clone(),
                            market_event_queue_info.clone(),
                            amm_authority_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            amm_target_orders_info.clone(),
                        ];
                        account_parser::PlanOrderBookArgs::with_parsed_args(
                            program_id,
                            spl_token_program_id,
                            monitor.plan_order_limit,
                            &mut amm,
                            plan_buy_accounts,
                            Self::do_plan_orderbook,
                        )
                        .unwrap();
                    }
                    AmmState::CancelOrderState => {
                        msg!("monitor_step CancelOrderState:{}", amm.state);
                        let cancel_order_accounts: &[AccountInfo] = &[
                            amm_info.clone(),
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_event_queue_info.clone(),
                            market_bids_info.clone(),
                            market_asks_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            amm_target_orders_info.clone(),
                        ];
                        account_parser::CancelOrderArgs::with_parsed_args(
                            program_id,
                            spl_token_program_id,
                            monitor.cancel_order_limit,
                            &mut amm,
                            cancel_order_accounts,
                            Self::do_cancel_order,
                        )
                        .unwrap();
                    }
                    AmmState::PlaceOrdersState => {
                        msg!("monitor_step PlaceOrdersState:{}", amm.state);
                        let place_order_accounts: &[AccountInfo] = &[
                            amm_info.clone(),
                            amm_authority_info.clone(),
                            amm_open_orders_info.clone(),
                            market_program_info.clone(),
                            market_info.clone(),
                            market_request_queue_info.clone(),
                            amm_coin_vault_info.clone(),
                            amm_pc_vault_info.clone(),
                            market_coin_vault_info.clone(),
                            market_pc_vault_info.clone(),
                            token_program_info.clone(),
                            rent_info.clone(),
                            market_event_queue_info.clone(),
                            market_bids_info.clone(),
                            market_asks_info.clone(),
                            amm_target_orders_info.clone(),
                            clock_info.clone(),
                        ];
                        account_parser::PlaceOrdersArgs::with_parsed_args(
                            program_id,
                            spl_token_program_id,
                            monitor.place_order_limit,
                            &mut amm,
                            place_order_accounts,
                            srm_token_info,
                            Self::do_place_orders,
                        )
                        .unwrap();
                    }
                    AmmState::PurgeOrderState => {
                        msg!("monitor_step PurgeOrderState:{}", amm.state);
                        let purge_orders_accounts: &[AccountInfo] = &[
                            amm_info.clone(),
                            market_program_info.clone(),
                            market_info.clone(),
                            amm_open_orders_info.clone(),
                            amm_authority_info.clone(),
                            market_event_queue_info.clone(),
                            market_bids_info.clone(),
                            market_asks_info.clone(),
                            amm_target_orders_info.clone(),
                        ];
                        account_parser::PurgeOrderArgs::with_parsed_args(
                            program_id,
                            spl_token_program_id,
                            monitor.cancel_order_limit,
                            &mut amm,
                            purge_orders_accounts,
                            Self::do_purge_orders,
                        )
                        .unwrap();
                    }
                    _ => {
                        msg!("monitor_step InvalidState:{}", amm.state);
                    }
                },
            }
        }
        msg!("monitor_step end");
        Ok(())
    }

    fn do_cancel_amm_orders<'a>(
        amm: &AmmInfo,
        amm_authority_info: &AccountInfo<'a>,
        amm_open_orders_info: &AccountInfo<'a>,
        market_program_info: &AccountInfo<'a>,
        market_info: &AccountInfo<'a>,
        market_bids_info: &AccountInfo<'a>,
        market_asks_info: &AccountInfo<'a>,
        market_event_queue_info: &AccountInfo<'a>,
        amm_seed: &[u8],
    ) -> ProgramResult {
        let (market_state, open_orders) = Processor::load_serum_market_order(
            market_info,
            amm_open_orders_info,
            amm_authority_info,
            amm,
            false,
        )?;
        let bids_orders = market_state.load_bids_mut(market_bids_info)?;
        let asks_orders = market_state.load_asks_mut(market_asks_info)?;
        let (bids, asks) = Processor::get_amm_orders(&open_orders, bids_orders, asks_orders)?;

        let mut amm_order_ids_vec = Vec::new();
        let mut order_ids = [0u64; 8];
        let mut count = 0;
        for i in 0..std::cmp::max(bids.len(), asks.len()) {
            if i < bids.len() {
                order_ids[count] = bids[i].client_order_id();
                count += 1;
            }
            if i < asks.len() {
                order_ids[count] = asks[i].client_order_id();
                count += 1;
            }
            if count == 8 {
                amm_order_ids_vec.push(order_ids);
                order_ids = [0u64; 8];
                count = 0;
            }
        }
        if count != 0 {
            amm_order_ids_vec.push(order_ids);
        }
        for ids in amm_order_ids_vec.iter() {
            Invokers::invoke_dex_cancel_orders_by_client_order_ids(
                market_program_info.clone(),
                market_info.clone(),
                market_bids_info.clone(),
                market_asks_info.clone(),
                amm_open_orders_info.clone(),
                amm_authority_info.clone(),
                market_event_queue_info.clone(),
                amm_seed,
                amm.nonce as u8,
                *ids,
            )?;
        }

        Ok(())
    }

    pub fn process_admin_cancel_orders(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        _cancel: AdminCancelOrdersInstruction,
    ) -> ProgramResult {
        const MIN_ACCOUNTS: usize = 17;
        let input_account_len = accounts.len();
        if input_account_len != MIN_ACCOUNTS
            && input_account_len != MIN_ACCOUNTS + 1
            && input_account_len != MIN_ACCOUNTS + 2
        {
            return Err(AmmError::WrongAccountsNumber.into());
        }

        let account_info_iter = &mut accounts.iter();
        let token_program_info = next_account_info(account_info_iter)?;

        let amm_info = next_account_info(account_info_iter)?;
        let amm_authority_info = next_account_info(account_info_iter)?;
        let amm_open_orders_info = next_account_info(account_info_iter)?;
        let amm_target_orders_info = next_account_info(account_info_iter)?;
        let amm_coin_vault_info = next_account_info(account_info_iter)?;
        let amm_pc_vault_info = next_account_info(account_info_iter)?;
        let amm_owner_info = next_account_info(account_info_iter)?;
        let amm_config_info = next_account_info(account_info_iter)?;

        let market_program_info = next_account_info(account_info_iter)?;
        let market_info = next_account_info(account_info_iter)?;
        let market_coin_vault_info = next_account_info(account_info_iter)?;
        let market_pc_vault_info = next_account_info(account_info_iter)?;
        let market_vault_signer_info = next_account_info(account_info_iter)?;
        let market_event_queue_info = next_account_info(account_info_iter)?;
        let market_bids_info = next_account_info(account_info_iter)?;
        let market_asks_info = next_account_info(account_info_iter)?;

        let mut srm_token_account = None;
        let mut referrer_pc_wallet = None;
        if input_account_len == MIN_ACCOUNTS + 1 {
            let srm_token_info = next_account_info(account_info_iter)?;
            srm_token_account = Some(srm_token_info);
        } else if input_account_len == MIN_ACCOUNTS + 2 {
            let srm_token_info = next_account_info(account_info_iter)?;
            let referrer_pc_info = next_account_info(account_info_iter)?;
            srm_token_account = Some(srm_token_info);
            referrer_pc_wallet = Some(referrer_pc_info);
        }

        if *token_program_info.key != spl_token::ID {
            return Err(AmmError::InvalidSplTokenProgram.into());
        }
        let amm = AmmInfo::load_checked(&amm_info, program_id)?;
        if *amm_authority_info.key
            != Self::authority_id(program_id, AUTHORITY_AMM, amm.nonce as u8)?
        {
            return Err(AmmError::InvalidProgramAddress.into());
        }
        if amm_info.owner != program_id {
            return Err(AmmError::InvalidOwner.into());
        }
        let (pda, _) = Pubkey::find_program_address(&[&AMM_CONFIG_SEED], program_id);
        if pda != *amm_config_info.key {
            return Err(AmmError::InvalidConfigAccount.into());
        }
        let amm_config = AmmConfig::load_checked(&amm_config_info, program_id)?;
        if !amm_owner_info.is_signer
            || (*amm_owner_info.key != config_feature::amm_owner::ID
                && *amm_owner_info.key != amm_config.cancel_owner)
        {
            return Err(AmmError::InvalidSignAccount.into());
        }

        if amm.status == AmmStatus::Uninitialized.into_u64() {
            return Err(AmmError::InvalidStatus.into());
        }
        if *market_program_info.key != amm.market_program {
            return Err(AmmError::InvalidMarketProgram.into());
        }
        if *market_info.key != amm.market {
            return Err(AmmError::InvalidMarket.into());
        }
        if *amm_open_orders_info.key != amm.open_orders {
            return Err(AmmError::InvalidOpenOrders.into());
        }
        if amm.coin_vault != *amm_coin_vault_info.key {
            return Err(AmmError::InvalidCoinVault.into());
        }
        if amm.pc_vault != *amm_pc_vault_info.key {
            return Err(AmmError::InvalidPCVault.into());
        }
        if amm.target_orders != *amm_target_orders_info.key {
            return Err(AmmError::InvalidTargetOrders.into());
        }
        if let Some(srm_token_info) = srm_token_account {
            let srm_token = Self::unpack_token_account(&srm_token_info, &spl_token::id())?;
            if *amm_authority_info.key != srm_token.owner {
                return Err(AmmError::InvalidOwner.into());
            }
            if srm_token.mint != srm_token::ID && srm_token.mint != msrm_token::ID {
                return Err(AmmError::InvalidSrmMint.into());
            }
        }
        if let Some(referrer_pc_info) = referrer_pc_wallet {
            let referrer_pc_token =
                Self::unpack_token_account(&referrer_pc_info, &spl_token::id())?;
            if referrer_pc_token.owner != config_feature::referrer_pc_wallet::id() {
                return Err(AmmError::InvalidOwner.into());
            }
            if referrer_pc_token.mint != amm.pc_vault_mint {
                return Err(AmmError::InvalidReferPCMint.into());
            }
        }
        Self::do_cancel_amm_orders(
            &amm,
            amm_authority_info,
            amm_open_orders_info,
            market_program_info,
            market_info,
            market_bids_info,
            market_asks_info,
            market_event_queue_info,
            AUTHORITY_AMM,
        )?;
        let (open_order_coin_free, open_order_pc_free) = {
            let (_market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                true,
            )?;
            (open_orders.native_coin_free, open_orders.native_pc_free)
        };
        if open_order_coin_free == 0 && open_order_pc_free == 0 {
            return Ok(());
        }
        Invokers::invoke_dex_settle_funds(
            market_program_info.clone(),
            market_info.clone(),
            amm_open_orders_info.clone(),
            amm_authority_info.clone(),
            market_coin_vault_info.clone(),
            market_pc_vault_info.clone(),
            amm_coin_vault_info.clone(),
            amm_pc_vault_info.clone(),
            market_vault_signer_info.clone(),
            token_program_info.clone(),
            referrer_pc_wallet,
            AUTHORITY_AMM,
            amm.nonce as u8,
        )?;
        Ok(())
    }

    /// Processes `process_create_config` instruction.
    pub fn process_create_config(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let admin_info = next_account_info(account_info_iter)?;
        let amm_config_info = next_account_info(account_info_iter)?;
        let pnl_owner_info = next_account_info(account_info_iter)?;
        let system_program_info = next_account_info(account_info_iter)?;
        let rent_sysvar_info = next_account_info(account_info_iter)?;

        if !admin_info.is_signer || config_feature::amm_owner::id() != *admin_info.key {
            return Err(AmmError::InvalidSignAccount.into());
        }
        if *system_program_info.key != solana_program::system_program::id() {
            return Err(AmmError::InvalidSysProgramAddress.into());
        }

        let (pda, bump_seed) = Pubkey::find_program_address(&[&&AMM_CONFIG_SEED], program_id);
        if pda != *amm_config_info.key {
            return Err(AmmError::InvalidConfigAccount.into());
        }
        if amm_config_info.owner != system_program_info.key {
            return Err(AmmError::RepeatCreateConfigAccount.into());
        }
        let pda_signer_seeds: &[&[_]] = &[&AMM_CONFIG_SEED, &[bump_seed]];
        let rent = &Rent::from_account_info(rent_sysvar_info)?;
        let data_size = size_of::<AmmConfig>();
        let required_lamports = rent
            .minimum_balance(data_size)
            .max(1)
            .saturating_sub(amm_config_info.lamports());
        if required_lamports > 0 {
            invoke(
                &system_instruction::transfer(
                    admin_info.key,
                    amm_config_info.key,
                    required_lamports,
                ),
                &[
                    admin_info.clone(),
                    amm_config_info.clone(),
                    system_program_info.clone(),
                ],
            )?;
        }
        invoke_signed(
            &system_instruction::allocate(amm_config_info.key, data_size as u64),
            &[amm_config_info.clone(), system_program_info.clone()],
            &[&pda_signer_seeds],
        )?;
        invoke_signed(
            &system_instruction::assign(amm_config_info.key, &program_id),
            &[amm_config_info.clone(), system_program_info.clone()],
            &[&pda_signer_seeds],
        )?;

        let mut amm_config = AmmConfig::load_mut_checked(&amm_config_info, program_id)?;
        amm_config.pnl_owner = *pnl_owner_info.key;
        amm_config.create_pool_fee = 0;

        Ok(())
    }

    /// Processes `process_update_config` instruction.
    pub fn process_update_config(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        config_args: ConfigArgs,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let admin_info = next_account_info(account_info_iter)?;
        let amm_config_info = next_account_info(account_info_iter)?;
        if !admin_info.is_signer || config_feature::amm_owner::id() != *admin_info.key {
            return Err(AmmError::InvalidSignAccount.into());
        }
        let (pda, _) = Pubkey::find_program_address(&[&AMM_CONFIG_SEED], program_id);
        if pda != *amm_config_info.key || amm_config_info.owner != program_id {
            return Err(AmmError::InvalidConfigAccount.into());
        }

        let mut amm_config = AmmConfig::load_mut_checked(&amm_config_info, program_id)?;
        match config_args.param {
            0 => {
                let pnl_owner = config_args.owner.unwrap();
                if pnl_owner == Pubkey::default() {
                    return Err(AmmError::InvalidInput.into());
                }
                amm_config.pnl_owner = pnl_owner;
            }
            1 => {
                let cancel_owner = config_args.owner.unwrap();
                if cancel_owner == Pubkey::default() {
                    return Err(AmmError::InvalidInput.into());
                }
                amm_config.cancel_owner = cancel_owner;
            }
            2 => {
                let create_pool_fee = config_args.create_pool_fee.unwrap();
                amm_config.create_pool_fee = create_pool_fee;
            }
            _ => {
                return Err(AmmError::InvalidInput.into());
            }
        }

        return Ok(());
    }

    /// Processes an [Instruction](enum.Instruction.html).
    pub fn process(program_id: &Pubkey, accounts: &[AccountInfo], input: &[u8]) -> ProgramResult {
        let instruction = AmmInstruction::unpack(input)?;
        match instruction {
            AmmInstruction::PreInitialize(_init_arg) => {
                unimplemented!("This instruction is not supported, please use Initialize2")
            }
            AmmInstruction::Initialize(_init1) => {
                unimplemented!("This instruction is not supported, please use Initialize2")
            }
            AmmInstruction::Initialize2(init2) => {
                Self::process_initialize2(program_id, accounts, init2)
            }
            AmmInstruction::MonitorStep(monitor) => {
                Self::process_monitor_step(program_id, accounts, monitor)
            }
            AmmInstruction::Deposit(deposit) => {
                Self::process_deposit(program_id, accounts, deposit)
            }
            AmmInstruction::Withdraw(withdraw) => {
                Self::process_withdraw(program_id, accounts, withdraw)
            }
            AmmInstruction::MigrateToOpenBook => {
                Self::process_migrate_to_openbook(program_id, accounts)
            }
            AmmInstruction::SetParams(setparams) => {
                Self::process_set_params(program_id, accounts, setparams)
            }
            AmmInstruction::WithdrawPnl => Self::process_withdrawpnl(program_id, accounts),
            AmmInstruction::WithdrawSrm(withdrawsrm) => {
                Self::process_withdraw_srm(program_id, accounts, withdrawsrm)
            }
            AmmInstruction::SwapBaseIn(swap) => {
                Self::process_swap_base_in(program_id, accounts, swap)
            }
            AmmInstruction::SwapBaseOut(swap) => {
                Self::process_swap_base_out(program_id, accounts, swap)
            }
            AmmInstruction::SimulateInfo(simulate) => {
                Self::process_simulate_info(program_id, accounts, simulate)
            }
            AmmInstruction::AdminCancelOrders(cancel) => {
                Self::process_admin_cancel_orders(program_id, accounts, cancel)
            }
            AmmInstruction::CreateConfigAccount => {
                Self::process_create_config(program_id, accounts)
            }
            AmmInstruction::UpdateConfigAccount(config_args) => {
                Self::process_update_config(program_id, accounts, config_args)
            }
        }
    }
}

pub mod account_parser {
    use super::*;
    pub struct IdleArgs<'a> {
        pub program_id: &'a Pubkey,
        pub total_coin_without_take_pnl: &'a mut u64,
        pub total_pc_without_take_pnl: &'a mut u64,
        pub amm: &'a mut AmmInfo,
        pub bids: &'a Vec<LeafNode>,
        pub asks: &'a Vec<LeafNode>,
        pub target: &'a mut TargetOrders,
    }
    impl<'a> IdleArgs<'a> {
        pub fn with_parsed_args(
            program_id: &'a Pubkey,
            spl_token_program_id: &'a Pubkey,
            amm: &mut AmmInfo,
            accounts: &'a [AccountInfo],
            f: impl FnOnce(IdleArgs) -> ProgramResult,
        ) -> ProgramResult {
            if accounts.len() != 11 {
                return Err(AmmError::WrongAccountsNumber.into());
            }
            let &[ref amm_info, ref _market_program_info, ref market_info, ref market_bids_info, ref market_asks_info, ref market_event_queue_info, ref market_authority_info, ref amm_open_orders_info, ref amm_coin_vault_info, ref amm_pc_vault_info, ref amm_target_orders_info] =
                array_ref![accounts, 0, 11];

            let amm_coin_vault =
                Processor::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
            let amm_pc_vault =
                Processor::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
            let mut target =
                TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;

            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                market_authority_info,
                &amm,
                false,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            let (bids, asks) = Processor::get_amm_orders(&open_orders, bids_orders, asks_orders)?;

            let (mut total_pc_without_take_pnl, mut total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;

            let args = IdleArgs {
                program_id,
                total_coin_without_take_pnl: &mut total_coin_without_take_pnl,
                total_pc_without_take_pnl: &mut total_pc_without_take_pnl,
                amm,
                bids: &bids,
                asks: &asks,
                target: &mut target,
            };
            f(args)
        }
    }

    pub struct CancelAllOrdersArgs<'a, 'b: 'a> {
        pub program_id: &'a Pubkey,
        pub limit: u16,
        pub market_program_info: &'a AccountInfo<'b>,
        pub market_info: &'a AccountInfo<'b>,
        pub amm_open_orders_info: &'a AccountInfo<'b>,
        pub amm_authority_info: &'a AccountInfo<'b>,
        pub market_event_queue_info: &'a AccountInfo<'b>,
        pub market_coin_vault_info: &'a AccountInfo<'b>,
        pub market_pc_vault_info: &'a AccountInfo<'b>,
        pub market_bids_info: &'a AccountInfo<'b>,
        pub market_asks_info: &'a AccountInfo<'b>,
        pub market_vault_signer: &'a AccountInfo<'b>,
        pub token_program_info: &'a AccountInfo<'b>,
        pub amm_coin_vault_info: &'a AccountInfo<'b>,
        pub amm_pc_vault_info: &'a AccountInfo<'b>,

        pub referrer_pc_account: Option<&'a AccountInfo<'b>>,

        pub amm: &'a mut AmmInfo,
        pub open_orders: &'a OpenOrders,
        pub target: &'a mut TargetOrders,
        pub bids: &'a Vec<LeafNode>,
        pub asks: &'a Vec<LeafNode>,
    }
    impl<'a, 'b: 'a> CancelAllOrdersArgs<'a, 'b> {
        pub fn with_parsed_args(
            program_id: &'a Pubkey,
            limit: u16,
            amm: &mut AmmInfo,
            accounts: &'a [AccountInfo<'b>],
            referrer_pc_wallet: Option<&'a AccountInfo<'b>>,
            f: impl FnOnce(CancelAllOrdersArgs) -> ProgramResult,
        ) -> ProgramResult {
            if accounts.len() != 15 {
                return Err(AmmError::WrongAccountsNumber.into());
            }
            let &[ref amm_info, ref market_program_info, ref market_info, ref amm_open_orders_info, ref amm_authority_info, ref market_event_queue_info, ref market_coin_vault_info, ref market_pc_vault_info, ref market_bids_info, ref market_asks_info, ref market_vault_signer, ref token_program_info, ref amm_coin_vault_info, ref amm_pc_vault_info, ref amm_target_orders_info] =
                array_ref![accounts, 0, 15];

            let mut target =
                TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                true,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            let (bids, asks) = Processor::get_amm_orders(&open_orders, bids_orders, asks_orders)?;

            let args = CancelAllOrdersArgs {
                program_id,
                limit,
                market_program_info,
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                market_event_queue_info,
                market_coin_vault_info,
                market_pc_vault_info,
                market_bids_info,
                market_asks_info,
                market_vault_signer,
                token_program_info,
                amm_coin_vault_info,
                amm_pc_vault_info,
                referrer_pc_account: referrer_pc_wallet,
                amm,
                open_orders: &open_orders,
                target: &mut target,
                bids: &bids,
                asks: &asks,
            };
            f(args)
        }
    }

    pub struct PlanOrderBookArgs<'a> {
        pub program_id: &'a Pubkey,
        pub limit: u16,
        pub total_coin_without_take_pnl: u64,
        pub total_pc_without_take_pnl: u64,
        pub amm: &'a mut AmmInfo,
        pub target: &'a mut TargetOrders,
    }
    impl<'a> PlanOrderBookArgs<'a> {
        pub fn with_parsed_args(
            program_id: &'a Pubkey,
            spl_token_program_id: &'a Pubkey,
            limit: u16,
            amm: &mut AmmInfo,
            accounts: &'a [AccountInfo],
            f: impl FnOnce(PlanOrderBookArgs) -> ProgramResult,
        ) -> ProgramResult {
            if accounts.len() != 8 {
                return Err(AmmError::WrongAccountsNumber.into());
            }
            let &[ref amm_info, ref market_info, ref market_event_queue_info, ref amm_authority_info, ref amm_open_orders_info, ref amm_coin_vault_info, ref amm_pc_vault_info, ref amm_target_orders_info] =
                array_ref![accounts, 0, 8];

            let amm_coin_vault =
                Processor::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
            let amm_pc_vault =
                Processor::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
            let mut target =
                TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;
            let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;

            let args = PlanOrderBookArgs {
                program_id,
                limit,
                total_coin_without_take_pnl,
                total_pc_without_take_pnl,
                amm,
                target: &mut target,
            };
            f(args)
        }
    }

    pub struct PlaceOrdersArgs<'a, 'b: 'a> {
        pub program_id: &'a Pubkey,
        pub limit: u16,

        pub amm_authority_info: &'a AccountInfo<'b>,
        pub amm_open_orders_info: &'a AccountInfo<'b>,
        pub market_program_info: &'a AccountInfo<'b>,
        pub market_info: &'a AccountInfo<'b>,
        pub market_request_queue_info: &'a AccountInfo<'b>,
        pub amm_coin_vault_info: &'a AccountInfo<'b>,
        pub amm_pc_vault_info: &'a AccountInfo<'b>,
        pub market_coin_vault_info: &'a AccountInfo<'b>,
        pub market_pc_vault_info: &'a AccountInfo<'b>,
        pub token_program_info: &'a AccountInfo<'b>,
        pub rent_info: &'a AccountInfo<'b>,

        pub market_event_queue_info: &'a AccountInfo<'b>,
        pub market_bids_info: &'a AccountInfo<'b>,
        pub market_asks_info: &'a AccountInfo<'b>,

        pub srm_token_account: Option<&'a AccountInfo<'b>>,

        pub amm: &'a mut AmmInfo,
        pub open_orders: &'a OpenOrders,
        pub bids: &'a Vec<LeafNode>,
        pub asks: &'a Vec<LeafNode>,
        pub target: &'a mut TargetOrders,

        pub total_coin_without_take_pnl: u64,
        pub total_pc_without_take_pnl: u64,
        pub coin_vault_amount: u64,
        pub pc_vault_amount: u64,
    }
    impl<'a, 'b: 'a> PlaceOrdersArgs<'a, 'b> {
        pub fn with_parsed_args(
            program_id: &'a Pubkey,
            spl_token_program_id: &'a Pubkey,
            limit: u16,
            amm: &mut AmmInfo,
            accounts: &'a [AccountInfo<'b>],
            srm_token_account: Option<&'a AccountInfo<'b>>,
            f: impl FnOnce(PlaceOrdersArgs) -> ProgramResult,
        ) -> ProgramResult {
            if accounts.len() != 17 {
                return Err(AmmError::WrongAccountsNumber.into());
            }
            let accounts = array_ref![accounts, 0, 17];
            let (new_orders_accounts, data_accounts) = array_refs![accounts, 15, 2];
            let &[ref amm_info, ref amm_authority_info, ref amm_open_orders_info, ref market_program_info, ref market_info, ref market_request_queue_info, ref amm_coin_vault_info, ref amm_pc_vault_info, ref market_coin_vault_info, ref market_pc_vault_info, ref token_program_info, ref rent_info, ref market_event_queue_info, ref market_bids_info, ref market_asks_info] =
                array_ref![new_orders_accounts, 0, 15];
            let &[ref amm_target_orders_info, ref _clock_info] = array_ref![data_accounts, 0, 2];

            let amm_coin_vault =
                Processor::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
            let amm_pc_vault =
                Processor::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
            let mut target =
                TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                false,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            let (bids, asks) = Processor::get_amm_orders(&open_orders, bids_orders, asks_orders)?;
            let (total_pc_without_take_pnl, total_coin_without_take_pnl) =
                Calculator::calc_total_without_take_pnl(
                    amm_pc_vault.amount,
                    amm_coin_vault.amount,
                    &open_orders,
                    &amm,
                    &market_state,
                    &market_event_queue_info,
                    &amm_open_orders_info,
                )?;

            let args = PlaceOrdersArgs {
                program_id,
                limit,

                amm_authority_info,
                amm_open_orders_info,
                market_program_info,
                market_info,
                market_request_queue_info,
                amm_coin_vault_info,
                amm_pc_vault_info,
                market_coin_vault_info,
                market_pc_vault_info,
                token_program_info,
                rent_info,

                market_event_queue_info,
                market_bids_info,
                market_asks_info,
                srm_token_account,

                amm,
                open_orders: &open_orders,
                bids: &bids,
                asks: &asks,
                target: &mut target,

                total_pc_without_take_pnl,
                total_coin_without_take_pnl,

                coin_vault_amount: amm_coin_vault.amount,
                pc_vault_amount: amm_pc_vault.amount,
            };
            f(args)
        }
    }

    pub struct PurgeOrderArgs<'a, 'b: 'a> {
        pub program_id: &'a Pubkey,
        pub limit: u16,
        pub market_program_info: &'a AccountInfo<'b>,
        pub market_info: &'a AccountInfo<'b>,
        pub amm_open_orders_info: &'a AccountInfo<'b>,
        pub amm_authority_info: &'a AccountInfo<'b>,
        pub market_event_queue_info: &'a AccountInfo<'b>,
        pub market_bids_info: &'a AccountInfo<'b>,
        pub market_asks_info: &'a AccountInfo<'b>,
        pub amm: &'a mut AmmInfo,
        pub target: &'a TargetOrders,
        pub bids: &'a Vec<LeafNode>,
        pub asks: &'a Vec<LeafNode>,
    }
    impl<'a, 'b: 'a> PurgeOrderArgs<'a, 'b> {
        pub fn with_parsed_args(
            program_id: &'a Pubkey,
            _spl_token_program_id: &'a Pubkey,
            limit: u16,
            amm: &mut AmmInfo,
            accounts: &'a [AccountInfo<'b>],
            f: impl FnOnce(PurgeOrderArgs) -> ProgramResult,
        ) -> ProgramResult {
            if accounts.len() != 9 {
                return Err(AmmError::WrongAccountsNumber.into());
            }
            let &[ref amm_info, ref market_program_info, ref market_info, ref amm_open_orders_info, ref amm_authority_info, ref market_event_queue_info, ref market_bids_info, ref market_asks_info, ref amm_target_orders_info] =
                array_ref![accounts, 0, 9];

            let target =
                TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                true,
            )?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            let (bids, asks) = Processor::get_amm_orders(&open_orders, bids_orders, asks_orders)?;

            let args = PurgeOrderArgs {
                program_id,
                limit,
                market_program_info,
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                market_event_queue_info,
                market_bids_info,
                market_asks_info,
                amm,
                target: &target,
                bids: &bids,
                asks: &asks,
            };
            f(args)
        }
    }

    pub struct CancelOrderArgs<'a, 'b: 'a> {
        pub program_id: &'a Pubkey,
        pub limit: u16,
        pub market_program_info: &'a AccountInfo<'b>,
        pub market_info: &'a AccountInfo<'b>,
        pub amm_open_orders_info: &'a AccountInfo<'b>,
        pub amm_authority_info: &'a AccountInfo<'b>,
        pub market_event_queue_info: &'a AccountInfo<'b>,
        pub market_bids_info: &'a AccountInfo<'b>,
        pub market_asks_info: &'a AccountInfo<'b>,
        pub amm: &'a mut AmmInfo,
        pub open_orders: &'a OpenOrders,
        pub target: &'a mut TargetOrders,
        pub bids: &'a Vec<LeafNode>,
        pub asks: &'a Vec<LeafNode>,

        pub coin_amount: u64,
        pub pc_amount: u64,
    }
    impl<'a, 'b: 'a> CancelOrderArgs<'a, 'b> {
        pub fn with_parsed_args(
            program_id: &'a Pubkey,
            spl_token_program_id: &'a Pubkey,
            limit: u16,
            amm: &mut AmmInfo,
            accounts: &'a [AccountInfo<'b>],
            f: impl FnOnce(CancelOrderArgs) -> ProgramResult,
        ) -> ProgramResult {
            if accounts.len() != 11 {
                return Err(AmmError::WrongAccountsNumber.into());
            }
            let &[ref amm_info, ref market_program_info, ref market_info, ref amm_open_orders_info, ref amm_authority_info, ref market_event_queue_info, ref market_bids_info, ref market_asks_info, ref amm_coin_vault_info, ref amm_pc_vault_info, ref amm_target_orders_info] =
                array_ref![accounts, 0, 11];

            let amm_coin_vault =
                Processor::unpack_token_account(&amm_coin_vault_info, spl_token_program_id)?;
            let amm_pc_vault =
                Processor::unpack_token_account(&amm_pc_vault_info, spl_token_program_id)?;
            let mut target =
                TargetOrders::load_mut_checked(&amm_target_orders_info, program_id, amm_info.key)?;
            let (market_state, open_orders) = Processor::load_serum_market_order(
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                &amm,
                true,
            )?;
            // let (myorders, _max_bid, _min_ask) = Processor::get_amm_orders_and_dex_best_price(&market_state, &open_orders, bids_acc, asks_acc)?;
            let bids_orders = market_state.load_bids_mut(&market_bids_info)?;
            let asks_orders = market_state.load_asks_mut(&market_asks_info)?;
            let (bids, asks) = Processor::get_amm_orders(&open_orders, bids_orders, asks_orders)?;

            let args = CancelOrderArgs {
                program_id,
                limit,
                market_program_info,
                market_info,
                amm_open_orders_info,
                amm_authority_info,
                market_event_queue_info,
                market_bids_info,
                market_asks_info,
                amm,
                open_orders: &open_orders,
                target: &mut target,
                bids: &bids,
                asks: &asks,
                coin_amount: amm_coin_vault.amount,
                pc_amount: amm_pc_vault.amount,
            };
            f(args)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_calc_take_pnl() {
        let mut amm = AmmInfo::default();
        amm.initialize(0, 0, 2, 9, 1000000, 1).unwrap();
        let mut target = TargetOrders::default();
        target.calc_pnl_x = 900000000000000;
        target.calc_pnl_y = 150000000000000000000000000;

        let mut total_pc_without_take_pnl = 1343675125663;
        let mut total_coin_without_take_pnl = 117837534493793;
        let x1 = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y1 = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );

        let (delta_x, delta_y) = Processor::calc_take_pnl(
            &target,
            &mut amm,
            &mut total_pc_without_take_pnl,
            &mut total_coin_without_take_pnl,
            x1.as_u128().into(),
            y1.as_u128().into(),
        )
        .unwrap();
        println!("delta_x:{}, delta_y:{}", delta_x, delta_y);
    }

    #[test]
    fn test_calc_pnl_precision() {
        // init
        let mut amm = AmmInfo::default();
        let init_pc_amount = 5434000000u64;
        let init_coin_amount = 100000000000000u64;
        let liquidity = Calculator::to_u64(
            U128::from(init_pc_amount)
                .checked_mul(init_coin_amount.into())
                .unwrap()
                .integer_sqrt()
                .as_u128(),
        )
        .unwrap();
        amm.initialize(0, 0, 5, 9, 1000000000, 7803).unwrap();
        amm.lp_amount = liquidity;

        let x =
            Calculator::normalize_decimal_v2(5434000000, amm.pc_decimals, amm.sys_decimal_value);
        let y = Calculator::normalize_decimal_v2(
            100000000000000,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );
        let mut target = TargetOrders::default();
        target.calc_pnl_x = x.as_u128();
        target.calc_pnl_y = y.as_u128();
        println!(
             "init_pc_amount:{}, init_coin_amount:{}, liquidity:{}, sys_decimal_value:{}, calc_pnl_x:{}, calc_pnl_y:{}",
             init_pc_amount, init_coin_amount, liquidity, amm.sys_decimal_value, target.calc_pnl_x, target.calc_pnl_y
         );

        // withdraw
        let withdraw_lp = 2577470628u64;
        let mut total_pc_without_take_pnl = init_pc_amount;
        let mut total_coin_without_take_pnl = init_coin_amount;
        let x1 = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y1 = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );

        let (delta_x, delta_y) = Processor::calc_take_pnl(
            &target,
            &mut amm,
            &mut total_pc_without_take_pnl,
            &mut total_coin_without_take_pnl,
            x1.as_u128().into(),
            y1.as_u128().into(),
        )
        .unwrap();
        println!("delta_x:{}, delta_y:{}", delta_x, delta_y);
        // coin_amount / total_coin_amount = amount / lp_mint.supply => coin_amount = total_coin_amount * amount / pool_mint.supply
        let invariant = InvariantPool {
            token_input: withdraw_lp,
            token_total: amm.lp_amount,
        };
        let coin_amount = invariant
            .exchange_pool_to_token(total_coin_without_take_pnl, RoundDirection::Floor)
            .ok_or(AmmError::CalculationExRateFailure)
            .unwrap();
        let pc_amount = invariant
            .exchange_pool_to_token(total_pc_without_take_pnl, RoundDirection::Floor)
            .ok_or(AmmError::CalculationExRateFailure)
            .unwrap();

        amm.lp_amount = amm.lp_amount.checked_sub(withdraw_lp).unwrap();
        target.calc_pnl_x = x1
            .checked_sub(Calculator::normalize_decimal_v2(
                pc_amount,
                amm.pc_decimals,
                amm.sys_decimal_value,
            ))
            .unwrap()
            .checked_sub(U128::from(delta_x))
            .unwrap()
            .as_u128();
        target.calc_pnl_y = y1
            .checked_sub(Calculator::normalize_decimal_v2(
                coin_amount,
                amm.coin_decimals,
                amm.sys_decimal_value,
            ))
            .unwrap()
            .checked_sub(U128::from(delta_y))
            .unwrap()
            .as_u128();
        total_pc_without_take_pnl = total_pc_without_take_pnl.checked_sub(pc_amount).unwrap();
        total_coin_without_take_pnl = total_coin_without_take_pnl
            .checked_sub(coin_amount)
            .unwrap();
        println!(
             "withdraw calc_pnl_x:{}, calc_pnl_y:{}, total_pc_without_take_pnl:{}, total_coin_without_take_pnl:{}",
             target.calc_pnl_x, target.calc_pnl_y, total_pc_without_take_pnl, total_coin_without_take_pnl
         );

        // withdraw 2
        let x1 = Calculator::normalize_decimal_v2(
            total_pc_without_take_pnl,
            amm.pc_decimals,
            amm.sys_decimal_value,
        );
        let y1 = Calculator::normalize_decimal_v2(
            total_coin_without_take_pnl,
            amm.coin_decimals,
            amm.sys_decimal_value,
        );

        let (delta_x, delta_y) = Processor::calc_take_pnl(
            &target,
            &mut amm,
            &mut total_pc_without_take_pnl,
            &mut total_coin_without_take_pnl,
            x1.as_u128().into(),
            y1.as_u128().into(),
        )
        .unwrap();
        println!("delta_x:{}, delta_y:{}", delta_x, delta_y);
    }

    #[test]
    fn test_swap_base_in() {
        let amount_in = 212854295571_u64;
        let total_coin_without_take_pnl = 77043918330755_u64;
        let total_pc_without_take_pnl = 1511361338135_u64;

        let swap_direction = SwapDirection::Coin2PC;
        let mut amm = AmmInfo::default();
        amm.initialize(0, 0, 2, 9, 1000000, 1).unwrap();

        let swap_fee = U128::from(amount_in)
            .checked_mul(amm.fees.swap_fee_numerator.into())
            .unwrap()
            .checked_ceil_div(amm.fees.swap_fee_denominator.into())
            .unwrap()
            .0;

        let swap_in_after_deduct_fee = U128::from(amount_in).checked_sub(swap_fee).unwrap();
        let swap_amount_out = Calculator::swap_token_amount_base_in(
            swap_in_after_deduct_fee,
            total_pc_without_take_pnl.into(),
            total_coin_without_take_pnl.into(),
            swap_direction,
        )
        .as_u64();

        println!("swap_amount_out:{}", swap_amount_out);
    }
}
