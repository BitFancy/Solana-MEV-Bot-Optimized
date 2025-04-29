use std::cmp::min;

use crate::assert_eq_admin;
use crate::constants::{
    BASIS_POINT_MAX, BIN_ARRAY_BITMAP_SIZE, FEE_PRECISION, MAX_BIN_ID, MAX_FEE_RATE,
    MAX_FEE_UPDATE_WINDOW, MIN_BIN_ID,
};
use crate::instructions::update_fee_parameters::FeeParameter;
use crate::math::u128x128_math::Rounding;
use crate::math::u64x64_math::SCALE_OFFSET;
use crate::math::utils_math::{one, safe_mul_div_cast, safe_mul_shr_cast, safe_shl_div_cast};
use crate::state::action_access::get_lb_pair_type_access_validator;
use crate::state::bin::BinArray;
use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::parameters::{StaticParameters, VariableParameters};
use crate::{errors::LBError, math::safe_math::SafeMath};
use anchor_lang::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use ruint::aliases::{U1024, U256};
use std::ops::BitXor;
use std::ops::Shl;
use std::ops::Shr;

#[derive(Copy, Clone, Debug, PartialEq, Eq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
/// Type of the Pair. 0 = Permissionless, 1 = Permission. Putting 0 as permissionless for backward compatibility.
pub enum PairType {
    Permissionless,
    Permission,
}

pub struct LaunchPadParams {
    pub activation_slot: u64,
}

impl PairType {
    pub fn get_pair_default_launch_pad_params(&self) -> LaunchPadParams {
        match self {
            // The slot is unreachable. Therefore by default, the pair will be disabled until admin update the activation slot.
            Self::Permission => LaunchPadParams {
                activation_slot: u64::MAX,
            },
            // Activation slot is not used in permissionless pair. Therefore, default to 0.
            Self::Permissionless => LaunchPadParams { activation_slot: 0 },
        }
    }
}

#[derive(
    AnchorSerialize, AnchorDeserialize, Debug, PartialEq, Eq, IntoPrimitive, TryFromPrimitive,
)]
#[repr(u8)]
/// Pair status. 0 = Enabled, 1 = Disabled. Putting 0 as enabled for backward compatibility.
pub enum PairStatus {
    // Fully enabled.
    // Condition:
    // Permissionless: PairStatus::Enabled
    // Permission: PairStatus::Enabled and clock.slot > activation_slot
    Enabled,
    // Similar as emergency mode. User can only withdraw (Only outflow). Except whitelisted wallet still have full privileges.
    Disabled,
}

#[zero_copy]
#[derive(InitSpace, Default, Debug)]
pub struct ProtocolFee {
    pub amount_x: u64,
    pub amount_y: u64,
}

#[account(zero_copy)]
#[derive(InitSpace, Debug)]
pub struct LbPair {
    pub parameters: StaticParameters,
    pub v_parameters: VariableParameters,
    pub bump_seed: [u8; 1],
    /// Bin step signer seed
    pub bin_step_seed: [u8; 2],
    /// Type of the pair
    pub pair_type: u8,
    /// Active bin id
    pub active_id: i32,
    /// Bin step. Represent the price increment / decrement.
    pub bin_step: u16,
    /// Status of the pair. Check PairStatus enum.
    pub status: u8,
    pub require_base_factor_seed: u8,
    pub base_factor_seed: [u8; 2],
    pub _padding1: [u8; 2],
    /// Token X mint
    pub token_x_mint: Pubkey,
    /// Token Y mint
    pub token_y_mint: Pubkey,
    /// LB token X vault
    pub reserve_x: Pubkey,
    /// LB token Y vault
    pub reserve_y: Pubkey,
    /// Uncollected protocol fee
    pub protocol_fee: ProtocolFee,
    /// Protocol fee owner,
    pub fee_owner: Pubkey,
    /// Farming reward information
    pub reward_infos: [RewardInfo; 2], // TODO: Bug in anchor IDL parser when using InitSpace macro. Temp hardcode it. https://github.com/coral-xyz/anchor/issues/2556
    /// Oracle pubkey
    pub oracle: Pubkey,
    /// Packed initialized bin array state
    pub bin_array_bitmap: [u64; 16], // store default bin id from -512 to 511 (bin id from -35840 to 35840, price from 2.7e-16 to 3.6e15)
    /// Last time the pool fee parameter was updated
    pub last_updated_at: i64,
    /// Whitelisted wallet
    pub whitelisted_wallet: Pubkey,
    /// Address allowed to swap when the current slot is greater than or equal to the pre-activation slot. The pre-activation slot is calculated as `activation_slot - pre_activation_slot_duration`.
    pub pre_activation_swap_address: Pubkey,
    /// Base keypair. Only required for permission pair
    pub base_key: Pubkey,
    /// Slot to enable the pair. Only applicable for permission pair.
    pub activation_slot: u64,
    /// Number of slot before activation slot. Used to calculate pre-activation slot for pre_activation_swap_address
    pub pre_activation_slot_duration: u64,
    /// _padding2 is reclaimed free space from swap_cap_deactivate_slot and swap_cap_amount before, BE CAREFUL FOR TOMBSTONE WHEN REUSE !!
    pub _padding2: [u8; 8],
    /// Liquidity lock duration for positions which created before activate. Only applicable for permission pair.
    pub lock_durations_in_slot: u64,
    /// Pool creator
    pub creator: Pubkey,
    /// Reserved space for future use
    pub _reserved: [u8; 24],
}

impl Default for LbPair {
    fn default() -> Self {
        let LaunchPadParams { activation_slot } =
            PairType::Permissionless.get_pair_default_launch_pad_params();
        Self {
            active_id: 0,
            parameters: StaticParameters::default(),
            v_parameters: VariableParameters::default(),
            bump_seed: [0u8; 1],
            bin_step: 0,
            token_x_mint: Pubkey::default(),
            token_y_mint: Pubkey::default(),
            bin_step_seed: [0u8; 2],
            fee_owner: Pubkey::default(),
            protocol_fee: ProtocolFee::default(),
            reserve_x: Pubkey::default(),
            reserve_y: Pubkey::default(),
            reward_infos: [RewardInfo::default(); 2],
            oracle: Pubkey::default(),
            bin_array_bitmap: [0u64; 16],
            last_updated_at: 0,
            pair_type: PairType::Permissionless.into(),
            status: 0,
            whitelisted_wallet: Pubkey::default(),
            pre_activation_slot_duration: 0,
            pre_activation_swap_address: Pubkey::default(),
            base_key: Pubkey::default(),
            activation_slot,
            creator: Pubkey::default(),
            lock_durations_in_slot: 0,
            require_base_factor_seed: 0,
            base_factor_seed: [0u8; 2],
            _padding1: [0u8; 2],
            _padding2: [0u8; 8],
            _reserved: [0u8; 24],
        }
    }
}

/// Stores the state relevant for tracking liquidity mining rewards
#[zero_copy]
#[derive(InitSpace, Default, Debug, PartialEq)]
pub struct RewardInfo {
    /// Reward token mint.
    pub mint: Pubkey,
    /// Reward vault token account.
    pub vault: Pubkey,
    /// Authority account that allows to fund rewards
    pub funder: Pubkey,
    /// TODO check whether we need to store it in pool
    pub reward_duration: u64, // 8
    /// TODO check whether we need to store it in pool
    pub reward_duration_end: u64, // 8
    /// TODO check whether we need to store it in pool
    pub reward_rate: u128, // 8
    /// The last time reward states were updated.
    pub last_update_time: u64, // 8
    /// Accumulated seconds where when farm distribute rewards, but the bin is empty. The reward will be accumulated for next reward time window.
    pub cumulative_seconds_with_empty_liquidity_reward: u64,
}

impl RewardInfo {
    /// Returns true if this reward is initialized.
    /// Once initialized, a reward cannot transition back to uninitialized.
    pub fn initialized(&self) -> bool {
        self.mint.ne(&Pubkey::default())
    }

    pub fn is_valid_funder(&self, funder: Pubkey) -> bool {
        assert_eq_admin(funder) || funder.eq(&self.funder)
    }

    pub fn init_reward(
        &mut self,
        mint: Pubkey,
        vault: Pubkey,
        funder: Pubkey,
        reward_duration: u64,
    ) {
        self.mint = mint;
        self.vault = vault;
        self.funder = funder;
        self.reward_duration = reward_duration;
    }

    pub fn update_last_update_time(&mut self, current_time: u64) {
        self.last_update_time = std::cmp::min(current_time, self.reward_duration_end);
    }

    pub fn get_seconds_elapsed_since_last_update(&self, current_time: u64) -> Result<u64> {
        let last_time_reward_applicable = std::cmp::min(current_time, self.reward_duration_end);
        let time_period = last_time_reward_applicable.safe_sub(self.last_update_time.into())?;

        Ok(time_period)
    }

    // To make it simple we truncate decimals of liquidity_supply for the calculation
    pub fn calculate_reward_per_token_stored_since_last_update(
        &self,
        current_time: u64,
        liquidity_supply: u64,
    ) -> Result<u128> {
        let time_period = self.get_seconds_elapsed_since_last_update(current_time)?;

        safe_mul_div_cast(
            time_period.into(),
            self.reward_rate,
            liquidity_supply.into(),
            Rounding::Down,
        )
    }

    pub fn calculate_reward_accumulated_since_last_update(
        &self,
        current_time: u64,
    ) -> Result<U256> {
        let last_time_reward_applicable = std::cmp::min(current_time, self.reward_duration_end);

        let time_period =
            U256::from(last_time_reward_applicable.safe_sub(self.last_update_time.into())?);

        Ok(time_period.safe_mul(U256::from(self.reward_rate))?)
    }

    /// Farming rate after funding
    pub fn update_rate_after_funding(
        &mut self,
        current_time: u64,
        funding_amount: u64,
    ) -> Result<()> {
        let reward_duration_end = self.reward_duration_end;
        let total_amount: u64;

        if current_time >= reward_duration_end {
            total_amount = funding_amount
        } else {
            let remaining_seconds = reward_duration_end.safe_sub(current_time)?;
            let leftover: u64 = safe_mul_shr_cast(
                self.reward_rate,
                remaining_seconds.into(),
                SCALE_OFFSET,
                Rounding::Down,
            )?;

            total_amount = leftover.safe_add(funding_amount)?;
        }

        self.reward_rate = safe_shl_div_cast(
            total_amount.into(),
            self.reward_duration.into(),
            SCALE_OFFSET,
            Rounding::Down,
        )?;
        self.last_update_time = current_time;
        self.reward_duration_end = current_time.safe_add(self.reward_duration)?;

        Ok(())
    }
}

impl LbPair {
    pub fn initialize(
        &mut self,
        bump: u8,
        active_id: i32,
        bin_step: u16,
        token_mint_x: Pubkey,
        token_mint_y: Pubkey,
        reserve_x: Pubkey,
        reserve_y: Pubkey,
        oracle: Pubkey,
        static_parameter: StaticParameters,
        pair_type: PairType,
        pair_status: u8,
        base_key: Pubkey,
        lock_duration_in_slot: u64,
        creator: Pubkey,
    ) -> Result<()> {
        self.parameters = static_parameter;
        self.active_id = active_id;
        self.bin_step = bin_step;
        self.token_x_mint = token_mint_x;
        self.token_y_mint = token_mint_y;
        self.reserve_x = reserve_x;
        self.reserve_y = reserve_y;
        self.fee_owner = crate::fee_owner::ID;
        self.bump_seed = [bump];
        self.bin_step_seed = bin_step.to_le_bytes();
        self.oracle = oracle;
        self.pair_type = pair_type.into();
        self.base_key = base_key;
        self.status = pair_status;
        self.creator = creator;

        let LaunchPadParams { activation_slot } = pair_type.get_pair_default_launch_pad_params();

        self.activation_slot = activation_slot;
        self.lock_durations_in_slot = lock_duration_in_slot;

        Ok(())
    }

    pub fn get_release_slot(&self, current_slot: u64) -> Result<u64> {
        let release_slot = match self.pair_type()? {
            PairType::Permission => {
                if self.lock_durations_in_slot > 0 && current_slot < self.activation_slot {
                    self.activation_slot
                        .saturating_add(self.lock_durations_in_slot)
                } else {
                    0
                }
            }
            _ => 0,
        };

        Ok(release_slot)
    }

    pub fn get_pre_activation_start_slot(&self) -> u64 {
        // pre_activation_slot_duration byte slice was swap_deactivation_slot before for permission pair, swap_deactivation_slot always >= activation_slot, so we need to do saturating_sub
        self.activation_slot
            .saturating_sub(self.pre_activation_slot_duration)
    }

    pub fn update_whitelisted_wallet(&mut self, wallet: Pubkey) -> Result<()> {
        self.whitelisted_wallet = wallet;
        Ok(())
    }

    pub fn status(&self) -> Result<PairStatus> {
        let pair_status: PairStatus = self
            .status
            .try_into()
            .map_err(|_| LBError::TypeCastFailed)?;

        Ok(pair_status)
    }

    pub fn pair_type(&self) -> Result<PairType> {
        let pair_type: PairType = self
            .pair_type
            .try_into()
            .map_err(|_| LBError::TypeCastFailed)?;

        Ok(pair_type)
    }

    pub fn is_permission_pair(&self) -> Result<bool> {
        let pair_type = self.pair_type()?;
        Ok(pair_type.eq(&PairType::Permission))
    }

    pub fn update_fee_parameters(&mut self, parameter: &FeeParameter) -> Result<()> {
        let current_timestamp = Clock::get()?.unix_timestamp;
        if self.last_updated_at > 0 {
            let second_elapsed = current_timestamp.safe_sub(self.last_updated_at)?;

            require!(
                second_elapsed > MAX_FEE_UPDATE_WINDOW,
                LBError::ExcessiveFeeUpdate
            );
        }

        self.parameters.update(parameter)?;
        self.last_updated_at = current_timestamp;

        Ok(())
    }

    pub fn seeds(&self) -> Result<Vec<&[u8]>> {
        let min_key = min(self.token_x_mint, self.token_y_mint);
        let (min_key_ref, max_key_ref) = if min_key == self.token_x_mint {
            (self.token_x_mint.as_ref(), self.token_y_mint.as_ref())
        } else {
            (self.token_y_mint.as_ref(), self.token_x_mint.as_ref())
        };
        if self.is_permission_pair()? {
            Ok(vec![
                self.base_key.as_ref(),
                min_key_ref,
                max_key_ref,
                &self.bin_step_seed,
                &self.bump_seed,
            ])
        } else {
            Ok(vec![
                min_key_ref,
                max_key_ref,
                &self.bin_step_seed,
                &self.bump_seed,
            ])
        }
    }

    #[inline(always)]
    pub fn swap_for_y(&self, out_token_mint: Pubkey) -> bool {
        out_token_mint.eq(&self.token_y_mint)
    }

    /// Plus / Minus 1 to the active bin based on the swap direction
    pub fn advance_active_bin(&mut self, swap_for_y: bool) -> Result<()> {
        let next_active_bin_id = if swap_for_y {
            self.active_id.safe_sub(1)?
        } else {
            self.active_id.safe_add(1)?
        };

        require!(
            next_active_bin_id >= MIN_BIN_ID && next_active_bin_id <= MAX_BIN_ID,
            LBError::PairInsufficientLiquidity
        );

        self.active_id = next_active_bin_id;

        Ok(())
    }

    /// Base fee rate = Base fee factor * bin step. This is in 1e9 unit.
    pub fn get_base_fee(&self) -> Result<u128> {
        Ok(u128::from(self.parameters.base_factor)
            .safe_mul(self.bin_step.into())?
            // Make it to be the same as FEE_PRECISION defined for ceil_div later on.
            .safe_mul(10u128)?)
    }

    /// Variable fee rate = variable fee factor * (volatility_accumulator * bin_step)^2
    pub fn compute_variable_fee(&self, volatility_accumulator: u32) -> Result<u128> {
        if self.parameters.variable_fee_control > 0 {
            let volatility_accumulator: u128 = volatility_accumulator.into();
            let bin_step: u128 = self.bin_step.into();
            let variable_fee_control: u128 = self.parameters.variable_fee_control.into();

            let square_vfa_bin = volatility_accumulator
                .safe_mul(bin_step)?
                .checked_pow(2)
                .ok_or(LBError::MathOverflow)?;

            // Variable fee control, volatility accumulator, bin step are in basis point unit (10_000)
            // This is 1e20. Which > 1e9. Scale down it to 1e9 unit and ceiling the remaining.
            let v_fee = variable_fee_control.safe_mul(square_vfa_bin)?;

            let scaled_v_fee = v_fee.safe_add(99_999_999_999)?.safe_div(100_000_000_000)?;
            return Ok(scaled_v_fee);
        }

        Ok(0)
    }

    /// Variable fee rate = variable_fee_control * (variable_fee_accumulator * bin_step) ^ 2
    pub fn get_variable_fee(&self) -> Result<u128> {
        self.compute_variable_fee(self.v_parameters.volatility_accumulator)
    }

    /// Total fee rate = base_fee_rate + variable_fee_rate
    pub fn get_total_fee(&self) -> Result<u128> {
        let total_fee_rate = self.get_base_fee()?.safe_add(self.get_variable_fee()?)?;
        let total_fee_rate_cap = std::cmp::min(total_fee_rate, MAX_FEE_RATE.into());
        Ok(total_fee_rate_cap)
    }

    #[cfg(test)]
    /// Maximum fee rate
    fn get_max_total_fee(&self) -> Result<u128> {
        let max_total_fee_rate = self
            .get_base_fee()?
            .safe_add(self.compute_variable_fee(self.parameters.max_volatility_accumulator)?)?;

        let total_fee_rate_cap = std::cmp::min(max_total_fee_rate, MAX_FEE_RATE.into());
        Ok(total_fee_rate_cap)
    }

    /// Compute composition fee. Composition_fee = fee_amount * (1 + total fee rate)
    pub fn compute_composition_fee(&self, swap_amount: u64) -> Result<u64> {
        let total_fee_rate = self.get_total_fee()?;
        // total_fee_rate 1e9 unit
        let fee_amount = u128::from(swap_amount).safe_mul(total_fee_rate)?;
        let composition_fee =
            fee_amount.safe_mul(u128::from(FEE_PRECISION).safe_add(total_fee_rate)?)?;
        // 1e9 unit * 1e9 unit = 1e18, scale back
        let scaled_down_fee = composition_fee.safe_div(u128::from(FEE_PRECISION).pow(2))?;

        Ok(scaled_down_fee
            .try_into()
            .map_err(|_| LBError::TypeCastFailed)?)
    }

    /// Compute fee from amount, where fee is part of the amount. The result is ceil-ed.
    pub fn compute_fee_from_amount(&self, amount_with_fees: u64) -> Result<u64> {
        // total_fee_rate 1e9 unit
        let total_fee_rate = self.get_total_fee()?;
        // Ceil division
        let fee_amount = u128::from(amount_with_fees)
            .safe_mul(total_fee_rate)?
            .safe_add((FEE_PRECISION - 1).into())?;
        let scaled_down_fee = fee_amount.safe_div(FEE_PRECISION.into())?;

        Ok(scaled_down_fee
            .try_into()
            .map_err(|_| LBError::TypeCastFailed)?)
    }

    /// Compute fee for the amount. The fee is not part of the amount. This function is used when you do not know the amount_with_fees
    /// Solve for fee_amount, equation: (amount + fee_amount) * total_fee_rate / 1e9 = fee_amount
    /// fee_amount = (amount * total_fee_rate) / (1e9 - total_fee_rate)
    /// The result is ceil-ed.
    pub fn compute_fee(&self, amount: u64) -> Result<u64> {
        let total_fee_rate = self.get_total_fee()?;
        let denominator = u128::from(FEE_PRECISION).safe_sub(total_fee_rate)?;

        // Ceil division
        let fee = u128::from(amount)
            .safe_mul(total_fee_rate)?
            .safe_add(denominator)?
            .safe_sub(1)?;

        let scaled_down_fee = fee.safe_div(denominator)?;

        Ok(scaled_down_fee
            .try_into()
            .map_err(|_| LBError::TypeCastFailed)?)
    }

    /// Compute protocol fee
    pub fn compute_protocol_fee(&self, fee_amount: u64) -> Result<u64> {
        let protocol_fee = u128::from(fee_amount)
            .safe_mul(self.parameters.protocol_share.into())?
            .safe_div(BASIS_POINT_MAX as u128)?;

        Ok(protocol_fee
            .try_into()
            .map_err(|_| LBError::TypeCastFailed)?)
    }

    /// Accumulate protocol fee
    pub fn accumulate_protocol_fees(&mut self, fee_amount_x: u64, fee_amount_y: u64) -> Result<()> {
        self.protocol_fee.amount_x = self.protocol_fee.amount_x.safe_add(fee_amount_x)?;
        self.protocol_fee.amount_y = self.protocol_fee.amount_y.safe_add(fee_amount_y)?;

        Ok(())
    }

    /// Update volatility reference and accumulator
    pub fn update_volatility_parameters(&mut self, current_timestamp: i64) -> Result<()> {
        self.v_parameters.update_volatility_parameter(
            self.active_id,
            current_timestamp,
            &self.parameters,
        )
    }

    pub fn update_references(&mut self, current_timestamp: i64) -> Result<()> {
        self.v_parameters
            .update_references(self.active_id, current_timestamp, &self.parameters)
    }

    pub fn update_volatility_accumulator(&mut self) -> Result<()> {
        self.v_parameters
            .update_volatility_accumulator(self.active_id, &self.parameters)
    }

    pub fn withdraw_protocol_fee(&mut self, amount_x: u64, amount_y: u64) -> Result<()> {
        self.protocol_fee.amount_x = self.protocol_fee.amount_x.safe_sub(amount_x)?;
        self.protocol_fee.amount_y = self.protocol_fee.amount_y.safe_sub(amount_y)?;

        Ok(())
    }

    pub fn set_fee_owner(&mut self, fee_owner: Pubkey) {
        self.fee_owner = fee_owner;
    }

    pub fn oracle_initialized(&self) -> bool {
        self.oracle != Pubkey::default()
    }

    pub fn flip_bin_array_bit(
        &mut self,
        bin_array_bitmap_extension: &Option<AccountLoader<BinArrayBitmapExtension>>,
        bin_array_index: i32,
    ) -> Result<()> {
        if self.is_overflow_default_bin_array_bitmap(bin_array_index) {
            match bin_array_bitmap_extension {
                Some(bitmap_ext) => {
                    bitmap_ext.load_mut()?.flip_bin_array_bit(bin_array_index)?;
                }
                None => return Err(LBError::BitmapExtensionAccountIsNotProvided.into()),
            }
        } else {
            self.flip_bin_array_bit_internal(bin_array_index)?;
        }

        Ok(())
    }

    pub fn is_overflow_default_bin_array_bitmap(&self, bin_array_index: i32) -> bool {
        let (min_bitmap_id, max_bitmap_id) = LbPair::bitmap_range();
        bin_array_index > max_bitmap_id || bin_array_index < min_bitmap_id
    }

    pub fn bitmap_range() -> (i32, i32) {
        (-BIN_ARRAY_BITMAP_SIZE, BIN_ARRAY_BITMAP_SIZE - 1)
    }

    fn get_bin_array_offset(bin_array_index: i32) -> usize {
        (bin_array_index + BIN_ARRAY_BITMAP_SIZE) as usize
    }

    fn flip_bin_array_bit_internal(&mut self, bin_array_index: i32) -> Result<()> {
        let bin_array_offset = Self::get_bin_array_offset(bin_array_index);
        let bin_array_bitmap = U1024::from_limbs(self.bin_array_bitmap);
        let mask = one::<1024, 16>() << bin_array_offset;
        self.bin_array_bitmap = bin_array_bitmap.bitxor(mask).into_limbs();
        Ok(())
    }

    // return bin_array_index that it's liquidity is non-zero
    // if cannot find one, return false
    pub fn next_bin_array_index_with_liquidity_internal(
        &self,
        swap_for_y: bool,
        start_array_index: i32,
    ) -> Result<(i32, bool)> {
        let bin_array_bitmap = U1024::from_limbs(self.bin_array_bitmap);
        let array_offset: usize = Self::get_bin_array_offset(start_array_index);
        let (min_bitmap_id, max_bitmap_id) = LbPair::bitmap_range();
        if swap_for_y {
            let binmap_range: usize = max_bitmap_id
                .safe_sub(min_bitmap_id)?
                .try_into()
                .map_err(|_| LBError::TypeCastFailed)?;
            let offset_bit_map = bin_array_bitmap.shl(binmap_range.safe_sub(array_offset)?);

            if offset_bit_map.eq(&U1024::ZERO) {
                return Ok((min_bitmap_id.safe_sub(1)?, false));
            } else {
                let next_bit = offset_bit_map.leading_zeros();
                return Ok((start_array_index.safe_sub(next_bit as i32)?, true));
            }
        } else {
            let offset_bit_map = bin_array_bitmap.shr(array_offset);
            if offset_bit_map.eq(&U1024::ZERO) {
                return Ok((max_bitmap_id.safe_add(1)?, false));
            } else {
                let next_bit = offset_bit_map.trailing_zeros();
                return Ok((
                    start_array_index.checked_add(next_bit as i32).unwrap(),
                    true,
                ));
            };
        }
    }

    // shift active until non-zero liquidity bin_array_index
    fn shift_active_bin(&mut self, swap_for_y: bool, bin_array_index: i32) -> Result<()> {
        // update active id
        let (lower_bin_id, upper_bin_id) =
            BinArray::get_bin_array_lower_upper_bin_id(bin_array_index)?;

        if swap_for_y {
            self.active_id = upper_bin_id;
        } else {
            self.active_id = lower_bin_id;
        }
        Ok(())
    }

    fn next_bin_array_index_with_liquidity_from_extension(
        swap_for_y: bool,
        bin_array_index: i32,
        bin_array_bitmap_extension: &Option<AccountLoader<BinArrayBitmapExtension>>,
    ) -> Result<(i32, bool)> {
        match bin_array_bitmap_extension {
            Some(bitmap_ext) => {
                return Ok(bitmap_ext
                    .load()?
                    .next_bin_array_index_with_liquidity(swap_for_y, bin_array_index)?);
            }
            None => return Err(LBError::BitmapExtensionAccountIsNotProvided.into()),
        }
    }

    pub fn next_bin_array_index_from_internal_to_extension(
        &mut self,
        swap_for_y: bool,
        current_array_index: i32,
        start_array_index: i32,
        bin_array_bitmap_extension: &Option<AccountLoader<BinArrayBitmapExtension>>,
    ) -> Result<()> {
        let (bin_array_index, is_non_zero_liquidity_flag) =
            self.next_bin_array_index_with_liquidity_internal(swap_for_y, start_array_index)?;
        if is_non_zero_liquidity_flag {
            if current_array_index != bin_array_index {
                self.shift_active_bin(swap_for_y, bin_array_index)?;
            }
        } else {
            let (bin_array_index, _) = LbPair::next_bin_array_index_with_liquidity_from_extension(
                swap_for_y,
                bin_array_index,
                bin_array_bitmap_extension,
            )?;
            // no need to check for flag here, because if we cannot find the non-liquidity bin array id in the extension go from lb_pair state, then extension will return error
            if current_array_index != bin_array_index {
                self.shift_active_bin(swap_for_y, bin_array_index)?;
            }
        }
        Ok(())
    }

    pub fn next_bin_array_index_with_liquidity(
        &mut self,
        swap_for_y: bool,
        bin_array_bitmap_extension: &Option<AccountLoader<BinArrayBitmapExtension>>,
    ) -> Result<()> {
        let start_array_index = BinArray::bin_id_to_bin_array_index(self.active_id)?;

        if self.is_overflow_default_bin_array_bitmap(start_array_index) {
            let (bin_array_index, is_non_zero_liquidity_flag) =
                LbPair::next_bin_array_index_with_liquidity_from_extension(
                    swap_for_y,
                    start_array_index,
                    bin_array_bitmap_extension,
                )?;
            if is_non_zero_liquidity_flag {
                if start_array_index != bin_array_index {
                    self.shift_active_bin(swap_for_y, bin_array_index)?;
                }
            } else {
                self.next_bin_array_index_from_internal_to_extension(
                    swap_for_y,
                    start_array_index,
                    bin_array_index,
                    bin_array_bitmap_extension,
                )?;
            }
        } else {
            self.next_bin_array_index_from_internal_to_extension(
                swap_for_y,
                start_array_index,
                start_array_index,
                bin_array_bitmap_extension,
            )?;
        }
        Ok(())
    }
}
