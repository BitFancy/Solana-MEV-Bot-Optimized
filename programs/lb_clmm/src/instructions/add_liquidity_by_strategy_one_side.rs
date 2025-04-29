use super::add_liquidity_by_strategy::StrategyParameters;
use crate::errors::LBError;
use crate::math::weight_to_amounts::to_amount_ask_side;
use crate::math::weight_to_amounts::to_amount_bid_side;
use crate::to_weight_ascending_order;
use crate::to_weight_descending_order;
use crate::to_weight_spot_balanced;
use crate::ModifyLiquidityOneSide;
use crate::StrategyType;
use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug, Default)]
pub struct LiquidityParameterByStrategyOneSide {
    /// Amount of X token or Y token to deposit
    pub amount: u64,
    /// Active bin that integrator observe off-chain
    pub active_id: i32,
    /// max active bin slippage allowed
    pub max_active_bin_slippage: i32,
    /// strategy parameters
    pub strategy_parameters: StrategyParameters,
}

impl LiquidityParameterByStrategyOneSide {
    pub fn to_amounts_into_bin(
        &self,
        active_id: i32,
        bin_step: u16,
        deposit_for_y: bool,
    ) -> Result<Vec<(i32, u64)>> {
        let min_bin_id = self.strategy_parameters.min_bin_id;
        let max_bin_id = self.strategy_parameters.max_bin_id;

        let weights = match self.strategy_parameters.strategy_type {
            StrategyType::SpotOneSide => Some(to_weight_spot_balanced(
                self.strategy_parameters.min_bin_id,
                self.strategy_parameters.max_bin_id,
            )),
            StrategyType::CurveOneSide => {
                if deposit_for_y {
                    Some(to_weight_ascending_order(min_bin_id, max_bin_id))
                } else {
                    Some(to_weight_descending_order(min_bin_id, max_bin_id))
                }
            }
            StrategyType::BidAskOneSide => {
                if deposit_for_y {
                    Some(to_weight_descending_order(min_bin_id, max_bin_id))
                } else {
                    Some(to_weight_ascending_order(min_bin_id, max_bin_id))
                }
            }
            _ => None,
        }
        .ok_or(LBError::InvalidStrategyParameters)?;

        if deposit_for_y {
            to_amount_bid_side(active_id, self.amount, &weights)
        } else {
            to_amount_ask_side(active_id, self.amount, bin_step, &weights)
        }
    }
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidityOneSide<'info>>,
    liquidity_parameter: &LiquidityParameterByStrategyOneSide,
) -> Result<()> {
    Ok(())
}
