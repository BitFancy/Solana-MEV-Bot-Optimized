use crate::errors::LBError;
use crate::math::safe_math::SafeMath;
use crate::math::weight_to_amounts::{to_amount_ask_side, to_amount_bid_side, to_amount_both_side};
use crate::ModifyLiquidity;
use anchor_lang::prelude::*;

const DEFAULT_MIN_WEIGHT: u16 = 200;
const DEFAULT_MAX_WEIGHT: u16 = 2000;

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug, Default)]
pub struct LiquidityParameterByStrategy {
    /// Amount of X token to deposit
    pub amount_x: u64,
    /// Amount of Y token to deposit
    pub amount_y: u64,
    /// Active bin that integrator observe off-chain
    pub active_id: i32,
    /// max active bin slippage allowed
    pub max_active_bin_slippage: i32,
    /// strategy parameters
    pub strategy_parameters: StrategyParameters,
}

impl LiquidityParameterByStrategy {
    pub fn to_amounts_into_bin(
        &self,
        active_id: i32,
        bin_step: u16,
        amount_x_in_active_bin: u64,
        amount_y_in_active_bin: u64,
    ) -> Result<Vec<(i32, u64, u64)>> {
        let min_bin_id = self.strategy_parameters.min_bin_id;
        let max_bin_id = self.strategy_parameters.max_bin_id;

        match self.strategy_parameters.strategy_type {
            StrategyType::SpotImBalanced => {
                let mut amounts_in_bin = vec![];
                if min_bin_id <= active_id {
                    let weights = to_weight_spot_balanced(min_bin_id, active_id);
                    let amounts_into_bid_side =
                        to_amount_bid_side(active_id, self.amount_y, &weights)?;
                    for &(bin_id, amount) in amounts_into_bid_side.iter() {
                        amounts_in_bin.push((bin_id, 0, amount))
                    }
                }
                if active_id < max_bin_id {
                    let weights = to_weight_spot_balanced(active_id + 1, max_bin_id);
                    let amounts_into_ask_side =
                        to_amount_ask_side(active_id, self.amount_x, bin_step, &weights)?;

                    for &(bin_id, amount) in amounts_into_ask_side.iter() {
                        amounts_in_bin.push((bin_id, amount, 0))
                    }
                }
                Ok(amounts_in_bin)
            }
            StrategyType::CurveImBalanced => {
                let mut amounts_in_bin = vec![];
                if min_bin_id <= active_id {
                    let weights = to_weight_ascending_order(min_bin_id, active_id);
                    let amounts_into_bid_side =
                        to_amount_bid_side(active_id, self.amount_y, &weights)?;
                    for &(bin_id, amount) in amounts_into_bid_side.iter() {
                        amounts_in_bin.push((bin_id, 0, amount))
                    }
                }
                if active_id < max_bin_id {
                    let weights = to_weight_descending_order(active_id + 1, max_bin_id);
                    let amounts_into_ask_side =
                        to_amount_ask_side(active_id, self.amount_x, bin_step, &weights)?;

                    for &(bin_id, amount) in amounts_into_ask_side.iter() {
                        amounts_in_bin.push((bin_id, amount, 0))
                    }
                }
                Ok(amounts_in_bin)
            }
            StrategyType::BidAskImBalanced => {
                let mut amounts_in_bin = vec![];
                if min_bin_id <= active_id {
                    let weights = to_weight_descending_order(min_bin_id, active_id);
                    let amounts_into_bid_side =
                        to_amount_bid_side(active_id, self.amount_y, &weights)?;
                    for &(bin_id, amount) in amounts_into_bid_side.iter() {
                        amounts_in_bin.push((bin_id, 0, amount))
                    }
                }
                if active_id < max_bin_id {
                    let weights = to_weight_ascending_order(active_id + 1, max_bin_id);
                    let amounts_into_ask_side =
                        to_amount_ask_side(active_id, self.amount_x, bin_step, &weights)?;

                    for &(bin_id, amount) in amounts_into_ask_side.iter() {
                        amounts_in_bin.push((bin_id, amount, 0))
                    }
                }
                Ok(amounts_in_bin)
            }
            StrategyType::SpotBalanced => {
                let weights = to_weight_spot_balanced(min_bin_id, max_bin_id);
                to_amount_both_side(
                    active_id,
                    bin_step,
                    amount_x_in_active_bin,
                    amount_y_in_active_bin,
                    self.amount_x,
                    self.amount_y,
                    &weights,
                )
            }
            StrategyType::CurveBalanced => {
                let weights = to_weight_curve(min_bin_id, max_bin_id, active_id)?;
                to_amount_both_side(
                    active_id,
                    bin_step,
                    amount_x_in_active_bin,
                    amount_y_in_active_bin,
                    self.amount_x,
                    self.amount_y,
                    &weights,
                )
            }
            StrategyType::BidAskBalanced => {
                let weights = to_weight_bid_ask(min_bin_id, max_bin_id, active_id)?;
                to_amount_both_side(
                    active_id,
                    bin_step,
                    amount_x_in_active_bin,
                    amount_y_in_active_bin,
                    self.amount_x,
                    self.amount_y,
                    &weights,
                )
            }
            _ => Err(LBError::InvalidStrategyParameters.into()),
        }
    }
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
    liquidity_parameter: &LiquidityParameterByStrategy,
) -> Result<()> {
    Ok(())
}

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug)]
pub struct StrategyParameters {
    /// min bin id
    pub min_bin_id: i32,
    /// max bin id
    pub max_bin_id: i32,
    /// strategy type
    pub strategy_type: StrategyType,
    /// parameters
    pub parameteres: [u8; 64],
}

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug)]
pub enum StrategyType {
    // spot one side
    SpotOneSide,
    // curve one side
    CurveOneSide,
    // bidAsk one side
    BidAskOneSide,
    // spot both side, balanced deposit
    SpotBalanced,
    // curve both side, balanced deposit
    CurveBalanced,
    // bid ask both side, balanced deposit
    BidAskBalanced,
    // spot both side, imbalanced deposit
    SpotImBalanced,
    // curve both side, imbalanced deposit
    CurveImBalanced,
    // bid ask both side, imbalanced deposit
    BidAskImBalanced,
}

pub fn to_weight_spot_balanced(min_bin_id: i32, max_bin_id: i32) -> Vec<(i32, u16)> {
    let mut weights = vec![];
    for i in min_bin_id..=max_bin_id {
        weights.push((i, 1));
    }
    weights
}

pub fn to_weight_descending_order(min_bin_id: i32, max_bin_id: i32) -> Vec<(i32, u16)> {
    let mut weights = vec![];
    for i in min_bin_id..=max_bin_id {
        weights.push((i, (max_bin_id - i + 1) as u16));
    }
    weights
}

pub fn to_weight_ascending_order(min_bin_id: i32, max_bin_id: i32) -> Vec<(i32, u16)> {
    let mut weights = vec![];
    for i in min_bin_id..=max_bin_id {
        weights.push((i, (i - min_bin_id + 1) as u16));
    }
    weights
}

pub fn to_weight_curve(
    min_bin_id: i32,
    max_bin_id: i32,
    active_id: i32,
) -> Result<Vec<(i32, u16)>> {
    if active_id < min_bin_id || active_id > max_bin_id {
        return Err(LBError::InvalidStrategyParameters.into());
    }
    let max_weight = DEFAULT_MAX_WEIGHT;
    let min_weight = DEFAULT_MIN_WEIGHT;

    let diff_weight = max_weight.safe_sub(min_weight)?;
    let diff_min_weight = if active_id > min_bin_id {
        diff_weight.safe_div(active_id.safe_sub(min_bin_id)? as u16)?
    } else {
        0
    };
    let diff_max_weight = if max_bin_id > active_id {
        diff_weight.safe_div(max_bin_id.safe_sub(active_id)? as u16)?
    } else {
        0
    };

    let mut weights = vec![];
    for i in min_bin_id..=max_bin_id {
        if i < active_id {
            let delta_bin = (active_id - i) as u16;
            let weight = max_weight - delta_bin * diff_min_weight;
            weights.push((i, weight));
        } else if i > active_id {
            let delta_bin = (i - active_id) as u16;
            let weight = max_weight - delta_bin * diff_max_weight;
            weights.push((i, weight));
        } else {
            weights.push((i, max_weight));
        }
    }
    Ok(weights)
}

pub fn to_weight_bid_ask(
    min_bin_id: i32,
    max_bin_id: i32,
    active_id: i32,
) -> Result<Vec<(i32, u16)>> {
    if active_id < min_bin_id || active_id > max_bin_id {
        return Err(LBError::InvalidStrategyParameters.into());
    }

    let max_weight = DEFAULT_MAX_WEIGHT;
    let min_weight = DEFAULT_MIN_WEIGHT;

    let diff_weight = max_weight.safe_sub(min_weight)?;

    let diff_min_weight = if active_id > min_bin_id {
        diff_weight.safe_div(active_id.safe_sub(min_bin_id)? as u16)?
    } else {
        0
    };
    let diff_max_weight = if max_bin_id > active_id {
        diff_weight.safe_div(max_bin_id.safe_sub(active_id)? as u16)?
    } else {
        0
    };

    let mut weights = vec![];
    for i in min_bin_id..=max_bin_id {
        if i < active_id {
            let delta_bin = (active_id - i) as u16;
            let weight = min_weight + delta_bin * diff_min_weight;
            weights.push((i, weight));
        } else if i > active_id {
            let delta_bin = (i - active_id) as u16;
            let weight = min_weight + delta_bin * diff_max_weight;
            weights.push((i, weight));
        } else {
            weights.push((i, min_weight));
        }
    }
    Ok(weights)
}

impl StrategyParameters {
    pub fn validate_both_side(&self, active_id: i32) -> Result<()> {
        if active_id < self.min_bin_id || active_id > self.max_bin_id {
            Err(LBError::InvalidStrategyParameters.into())
        } else {
            Ok(())
        }
    }
    pub fn bin_count(&self) -> Result<usize> {
        let bin_count = self.max_bin_id.safe_sub(self.min_bin_id)?;
        Ok(bin_count as usize)
    }
}

impl Default for StrategyParameters {
    fn default() -> Self {
        StrategyParameters {
            min_bin_id: 0,
            max_bin_id: 0,
            strategy_type: StrategyType::SpotBalanced,
            parameteres: [0; 64],
        }
    }
}
