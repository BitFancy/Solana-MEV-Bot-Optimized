use super::bin::Bin;
use crate::{
    constants::{MAX_BIN_PER_POSITION, NUM_REWARDS},
    errors::LBError,
    manager::bin_array_manager::BinArrayManager,
    math::{
        safe_math::SafeMath, u128x128_math::Rounding, u64x64_math::SCALE_OFFSET,
        utils_math::safe_mul_shr_cast,
    },
};
use anchor_lang::prelude::*;
use num_traits::Zero;
use std::cell::Ref;

#[account(zero_copy)]
#[derive(InitSpace, Debug)]
pub struct Position {
    /// The LB pair of this position
    pub lb_pair: Pubkey,
    /// Owner of the position. Client rely on this to to fetch their positions.
    pub owner: Pubkey,
    /// Liquidity shares of this position in bins (lower_bin_id <-> upper_bin_id). This is the same as LP concept.
    pub liquidity_shares: [u64; MAX_BIN_PER_POSITION],
    /// Farming reward information
    pub reward_infos: [UserRewardInfo; MAX_BIN_PER_POSITION],
    /// Swap fee to claim information
    pub fee_infos: [FeeInfo; MAX_BIN_PER_POSITION],
    /// Lower bin ID
    pub lower_bin_id: i32,
    /// Upper bin ID
    pub upper_bin_id: i32,
    /// Last updated timestamp
    pub last_updated_at: i64,
    /// Total claimed token fee X
    pub total_claimed_fee_x_amount: u64,
    /// Total claimed token fee Y
    pub total_claimed_fee_y_amount: u64,
    /// Total claimed rewards
    pub total_claimed_rewards: [u64; 2],
    /// Reserved space for future use
    pub _reserved: [u8; 160],
}

#[account(zero_copy)]
#[derive(InitSpace, Debug)]
pub struct PositionV2 {
    /// The LB pair of this position
    pub lb_pair: Pubkey,
    /// Owner of the position. Client rely on this to to fetch their positions.
    pub owner: Pubkey,
    /// Liquidity shares of this position in bins (lower_bin_id <-> upper_bin_id). This is the same as LP concept.
    pub liquidity_shares: [u128; MAX_BIN_PER_POSITION],
    /// Farming reward information
    pub reward_infos: [UserRewardInfo; MAX_BIN_PER_POSITION],
    /// Swap fee to claim information
    pub fee_infos: [FeeInfo; MAX_BIN_PER_POSITION],
    /// Lower bin ID
    pub lower_bin_id: i32,
    /// Upper bin ID
    pub upper_bin_id: i32,
    /// Last updated timestamp
    pub last_updated_at: i64,
    /// Total claimed token fee X
    pub total_claimed_fee_x_amount: u64,
    /// Total claimed token fee Y
    pub total_claimed_fee_y_amount: u64,
    /// Total claimed rewards
    pub total_claimed_rewards: [u64; 2],
    /// Operator of position
    pub operator: Pubkey,
    /// Slot which the locked liquidity can be withdraw
    pub lock_release_slot: u64,
    /// Is the position subjected to liquidity locking for the launch pool.
    pub subjected_to_bootstrap_liquidity_locking: u8,
    /// Address is able to claim fee in this position, only valid for bootstrap_liquidity_position
    pub fee_owner: Pubkey,
    /// Reserved space for future use
    pub _reserved: [u8; 87],
}

impl Default for PositionV2 {
    fn default() -> Self {
        Self {
            lb_pair: Pubkey::default(),
            owner: Pubkey::default(),
            lower_bin_id: 0,
            upper_bin_id: 0,
            last_updated_at: 0,
            liquidity_shares: [0u128; MAX_BIN_PER_POSITION],
            reward_infos: [UserRewardInfo::default(); MAX_BIN_PER_POSITION],
            fee_infos: [FeeInfo::default(); MAX_BIN_PER_POSITION],
            total_claimed_fee_x_amount: 0,
            total_claimed_fee_y_amount: 0,
            total_claimed_rewards: [0u64; 2],
            operator: Pubkey::default(),
            subjected_to_bootstrap_liquidity_locking: 0,
            lock_release_slot: 0,
            fee_owner: Pubkey::default(),
            _reserved: [0u8; 87],
        }
    }
}

#[zero_copy]
#[derive(Default, Debug, AnchorDeserialize, AnchorSerialize, InitSpace, PartialEq)]
pub struct FeeInfo {
    pub fee_x_per_token_complete: u128,
    pub fee_y_per_token_complete: u128,
    pub fee_x_pending: u64,
    pub fee_y_pending: u64,
}

#[zero_copy]
#[derive(Default, Debug, AnchorDeserialize, AnchorSerialize, InitSpace, PartialEq)]
pub struct UserRewardInfo {
    pub reward_per_token_completes: [u128; NUM_REWARDS],
    pub reward_pendings: [u64; NUM_REWARDS],
}

impl PositionV2 {
    pub fn init(
        &mut self,
        lb_pair: Pubkey,
        owner: Pubkey,
        operator: Pubkey,
        lower_bin_id: i32,
        upper_bin_id: i32,
        current_time: i64,
        lock_release_slot: u64,
        subjected_to_bootstrap_liquidity_locking: bool,
        fee_owner: Pubkey,
    ) -> Result<()> {
        self.lb_pair = lb_pair;
        self.owner = owner;
        self.operator = operator;

        self.lower_bin_id = lower_bin_id;
        self.upper_bin_id = upper_bin_id;

        self.liquidity_shares = [0u128; MAX_BIN_PER_POSITION];
        self.reward_infos = [UserRewardInfo::default(); MAX_BIN_PER_POSITION];

        self.last_updated_at = current_time;
        self.lock_release_slot = lock_release_slot;
        self.subjected_to_bootstrap_liquidity_locking =
            subjected_to_bootstrap_liquidity_locking.into();

        if subjected_to_bootstrap_liquidity_locking {
            self.fee_owner = fee_owner;
        }

        Ok(())
    }

    pub fn migrate_from_v1(&mut self, position: Ref<'_, Position>) -> Result<()> {
        self.lb_pair = position.lb_pair;
        self.owner = position.owner;
        self.reward_infos = position.reward_infos;
        self.fee_infos = position.fee_infos;
        self.lower_bin_id = position.lower_bin_id;
        self.upper_bin_id = position.upper_bin_id;
        self.total_claimed_fee_x_amount = position.total_claimed_fee_x_amount;
        self.total_claimed_fee_y_amount = position.total_claimed_fee_y_amount;
        self.total_claimed_rewards = position.total_claimed_rewards;
        self.last_updated_at = position.last_updated_at;

        for (i, &liquidity_share) in position.liquidity_shares.iter().enumerate() {
            self.liquidity_shares[i] = u128::from(liquidity_share).safe_shl(SCALE_OFFSET.into())?;
        }
        Ok(())
    }

    pub fn id_within_position(&self, id: i32) -> Result<()> {
        require!(
            id >= self.lower_bin_id && id <= self.upper_bin_id,
            LBError::InvalidPosition
        );
        Ok(())
    }

    /// Return the width of the position. The width is 1 when the position have the same value for upper_bin_id, and lower_bin_id.
    pub fn width(&self) -> Result<i32> {
        Ok(self.upper_bin_id.safe_sub(self.lower_bin_id)?.safe_add(1)?)
    }

    pub fn get_idx(&self, bin_id: i32) -> Result<usize> {
        self.id_within_position(bin_id)?;
        Ok(bin_id.safe_sub(self.lower_bin_id)? as usize)
    }

    pub fn from_idx_to_bin_id(&self, i: usize) -> Result<i32> {
        Ok(self.lower_bin_id.safe_add(i as i32)?)
    }

    pub fn withdraw(&mut self, bin_id: i32, liquidity_share: u128) -> Result<()> {
        let idx = self.get_idx(bin_id)?;
        self.liquidity_shares[idx] = self.liquidity_shares[idx].safe_sub(liquidity_share)?;

        Ok(())
    }

    pub fn deposit(&mut self, bin_id: i32, liquidity_share: u128) -> Result<()> {
        let idx = self.get_idx(bin_id)?;
        self.liquidity_shares[idx] = self.liquidity_shares[idx].safe_add(liquidity_share)?;

        Ok(())
    }

    pub fn get_liquidity_share_in_bin(&self, bin_id: i32) -> Result<u128> {
        let idx = self.get_idx(bin_id)?;
        Ok(self.liquidity_shares[idx])
    }

    pub fn accumulate_total_claimed_rewards(&mut self, reward_index: usize, reward: u64) {
        let total_claimed_reward = self.total_claimed_rewards[reward_index];
        self.total_claimed_rewards[reward_index] = total_claimed_reward.wrapping_add(reward);
    }

    pub fn accumulate_total_claimed_fees(&mut self, fee_x: u64, fee_y: u64) {
        self.total_claimed_fee_x_amount = self.total_claimed_fee_x_amount.wrapping_add(fee_x);
        self.total_claimed_fee_y_amount = self.total_claimed_fee_y_amount.wrapping_add(fee_y);
    }

    /// Update reward + fee earning
    pub fn update_earning_per_token_stored(
        &mut self,
        bin_array_manager: &BinArrayManager,
    ) -> Result<()> {
        let (bin_arrays_lower_bin_id, bin_arrays_upper_bin_id) =
            bin_array_manager.get_lower_upper_bin_id()?;

        // Make sure that the bin arrays cover all the bins of the position.
        // TODO: Should we? Maybe we shall update only the bins the user are interacting with, and allow chunk for claim reward.
        require!(
            self.lower_bin_id >= bin_arrays_lower_bin_id
                && self.upper_bin_id <= bin_arrays_upper_bin_id,
            LBError::InvalidBinArray
        );

        for bin_id in self.lower_bin_id..=self.upper_bin_id {
            let bin = bin_array_manager.get_bin(bin_id)?;
            self.update_reward_per_token_stored(bin_id, &bin)?;
            self.update_fee_per_token_stored(bin_id, &bin)?;
        }

        Ok(())
    }

    pub fn update_fee_per_token_stored(&mut self, bin_id: i32, bin: &Bin) -> Result<()> {
        let idx = self.get_idx(bin_id)?;

        let fee_infos = &mut self.fee_infos[idx];

        let fee_x_per_token_stored = bin.fee_amount_x_per_token_stored;

        let new_fee_x: u64 = safe_mul_shr_cast(
            self.liquidity_shares[idx]
                .safe_shr(SCALE_OFFSET.into())?
                .try_into()
                .map_err(|_| LBError::TypeCastFailed)?,
            fee_x_per_token_stored.safe_sub(fee_infos.fee_x_per_token_complete)?,
            SCALE_OFFSET,
            Rounding::Down,
        )?;

        fee_infos.fee_x_pending = new_fee_x.safe_add(fee_infos.fee_x_pending)?;
        fee_infos.fee_x_per_token_complete = fee_x_per_token_stored;

        let fee_y_per_token_stored = bin.fee_amount_y_per_token_stored;

        let new_fee_y: u64 = safe_mul_shr_cast(
            self.liquidity_shares[idx]
                .safe_shr(SCALE_OFFSET.into())?
                .try_into()
                .map_err(|_| LBError::TypeCastFailed)?,
            fee_y_per_token_stored.safe_sub(fee_infos.fee_y_per_token_complete)?,
            SCALE_OFFSET,
            Rounding::Down,
        )?;

        fee_infos.fee_y_pending = new_fee_y.safe_add(fee_infos.fee_y_pending)?;
        fee_infos.fee_y_per_token_complete = fee_y_per_token_stored;

        Ok(())
    }

    pub fn update_reward_per_token_stored(&mut self, bin_id: i32, bin: &Bin) -> Result<()> {
        let idx = self.get_idx(bin_id)?;

        let reward_info = &mut self.reward_infos[idx];
        for reward_idx in 0..NUM_REWARDS {
            let reward_per_token_stored = bin.reward_per_token_stored[reward_idx];

            let new_reward: u64 = safe_mul_shr_cast(
                self.liquidity_shares[idx]
                    .safe_shr(SCALE_OFFSET.into())?
                    .try_into()
                    .map_err(|_| LBError::TypeCastFailed)?,
                reward_per_token_stored
                    .safe_sub(reward_info.reward_per_token_completes[reward_idx])?,
                SCALE_OFFSET,
                Rounding::Down,
            )?;

            reward_info.reward_pendings[reward_idx] =
                new_reward.safe_add(reward_info.reward_pendings[reward_idx])?;
            reward_info.reward_per_token_completes[reward_idx] = reward_per_token_stored;
        }

        Ok(())
    }

    pub fn get_total_reward(&self, reward_index: usize) -> Result<u64> {
        let mut total_reward = 0;
        for val in self.reward_infos.iter() {
            total_reward = total_reward.safe_add(val.reward_pendings[reward_index])?;
        }
        Ok(total_reward)
    }

    pub fn reset_all_pending_reward(&mut self, reward_index: usize) {
        for val in self.reward_infos.iter_mut() {
            val.reward_pendings[reward_index] = 0;
        }
    }

    pub fn claim_fee(&mut self) -> Result<(u64, u64)> {
        let mut fee_x = 0;
        let mut fee_y = 0;

        for fee_info in self.fee_infos.iter_mut() {
            fee_x = fee_x.safe_add(fee_info.fee_x_pending)?;
            fee_info.fee_x_pending = 0;

            fee_y = fee_y.safe_add(fee_info.fee_y_pending)?;
            fee_info.fee_y_pending = 0;
        }

        Ok((fee_x, fee_y))
    }

    pub fn set_last_updated_at(&mut self, current_time: i64) {
        self.last_updated_at = current_time;
    }

    /// Position is empty when rewards is 0, fees is 0, and liquidity share is 0.
    pub fn is_empty(&self) -> bool {
        for (idx, liquidity_share) in self.liquidity_shares.iter().enumerate() {
            if !liquidity_share.is_zero() {
                return false;
            }
            let reward_infos = &self.reward_infos[idx];

            for reward_pending in reward_infos.reward_pendings {
                if !reward_pending.is_zero() {
                    return false;
                }
            }

            let fee_infos = &self.fee_infos[idx];
            if !fee_infos.fee_x_pending.is_zero() || !fee_infos.fee_y_pending.is_zero() {
                return false;
            }
        }
        true
    }

    pub fn is_liquidity_locked(&self, current_slot: u64) -> bool {
        current_slot < self.lock_release_slot
    }

    pub fn is_subjected_to_initial_liquidity_locking(&self) -> bool {
        self.subjected_to_bootstrap_liquidity_locking != 0
    }
}
