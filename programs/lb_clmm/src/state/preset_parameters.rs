use crate::constants::{BASIS_POINT_MAX, MAX_PROTOCOL_SHARE, U24_MAX};
use crate::errors::LBError;
use crate::math::price_math::get_price_from_id;
use anchor_lang::prelude::*;

use super::parameters::StaticParameters;

#[account]
#[derive(InitSpace, Debug)]
pub struct PresetParameter {
    /// Bin step. Represent the price increment / decrement.
    pub bin_step: u16,
    /// Used for base fee calculation. base_fee_rate = base_factor * bin_step
    pub base_factor: u16,
    /// Filter period determine high frequency trading time window.
    pub filter_period: u16,
    /// Decay period determine when the volatile fee start decay / decrease.
    pub decay_period: u16,
    /// Reduction factor controls the volatile fee rate decrement rate.
    pub reduction_factor: u16,
    /// Used to scale the variable fee component depending on the dynamic of the market
    pub variable_fee_control: u32,
    /// Maximum number of bin crossed can be accumulated. Used to cap volatile fee rate.
    pub max_volatility_accumulator: u32,
    /// Min bin id supported by the pool based on the configured bin step.
    pub min_bin_id: i32,
    /// Max bin id supported by the pool based on the configured bin step.
    pub max_bin_id: i32,
    /// Portion of swap fees retained by the protocol by controlling protocol_share parameter. protocol_swap_fee = protocol_share * total_swap_fee
    pub protocol_share: u16,
}

impl PresetParameter {
    pub fn init(
        &mut self,
        bin_step: u16,
        base_factor: u16,
        filter_period: u16,
        decay_period: u16,
        reduction_factor: u16,
        variable_fee_control: u32,
        max_volatility_accumulator: u32,
        min_bin_id: i32,
        max_bin_id: i32,
        protocol_share: u16,
    ) {
        self.bin_step = bin_step;
        self.base_factor = base_factor;
        self.filter_period = filter_period;
        self.decay_period = decay_period;
        self.reduction_factor = reduction_factor;
        self.variable_fee_control = variable_fee_control;
        self.max_volatility_accumulator = max_volatility_accumulator;
        self.min_bin_id = min_bin_id;
        self.max_bin_id = max_bin_id;
        self.protocol_share = protocol_share;
    }

    pub fn update(
        &mut self,
        base_factor: u16,
        filter_period: u16,
        decay_period: u16,
        reduction_factor: u16,
        variable_fee_control: u32,
        max_volatility_accumulator: u32,
        protocol_share: u16,
    ) {
        self.init(
            self.bin_step,
            base_factor,
            filter_period,
            decay_period,
            reduction_factor,
            variable_fee_control,
            max_volatility_accumulator,
            self.min_bin_id,
            self.max_bin_id,
            protocol_share,
        );
    }

    pub fn validate(&self) -> Result<()> {
        require!(
            self.bin_step <= BASIS_POINT_MAX as u16,
            LBError::InvalidInput
        );

        // we don't rug
        require!(
            self.protocol_share <= MAX_PROTOCOL_SHARE,
            LBError::InvalidInput
        );

        // filter period < t < decay period
        require!(
            self.filter_period < self.decay_period,
            LBError::InvalidInput
        );

        // reduction factor decide the decay rate of variable fee, max reduction_factor is BASIS_POINT_MAX = 100% reduction
        require!(
            self.reduction_factor <= BASIS_POINT_MAX as u16,
            LBError::InvalidInput
        );

        // prevent program overflow
        require!(self.variable_fee_control <= U24_MAX, LBError::InvalidInput);
        require!(
            self.max_volatility_accumulator <= U24_MAX,
            LBError::InvalidInput
        );

        let max_price = get_price_from_id(self.max_bin_id, self.bin_step);
        let min_price = get_price_from_id(self.min_bin_id, self.bin_step);

        require!(max_price.is_ok(), LBError::InvalidInput);
        require!(min_price.is_ok(), LBError::InvalidInput);

        // Bin is not swap-able when the price is u128::MAX, and 1. Make sure the min and max price bound is 2**127 - 1, 2
        if let Ok(max_price) = max_price {
            require!(
                max_price == 170141183460469231731687303715884105727,
                LBError::InvalidInput
            );
        }

        if let Ok(min_price) = min_price {
            require!(min_price == 2, LBError::InvalidInput);
        }

        Ok(())
    }

    pub fn to_static_parameters(&self) -> StaticParameters {
        StaticParameters {
            base_factor: self.base_factor,
            decay_period: self.decay_period,
            filter_period: self.filter_period,
            max_bin_id: self.max_bin_id,
            min_bin_id: self.min_bin_id,
            variable_fee_control: self.variable_fee_control,
            reduction_factor: self.reduction_factor,
            protocol_share: self.protocol_share,
            max_volatility_accumulator: self.max_volatility_accumulator,
            _padding: [0u8; 6],
        }
    }
}
