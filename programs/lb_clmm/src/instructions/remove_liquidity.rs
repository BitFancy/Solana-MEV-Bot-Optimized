use super::add_liquidity::ModifyLiquidity;
use crate::constants::BASIS_POINT_MAX;
use crate::{errors::LBError, math::safe_math::SafeMath, state::position::PositionV2};
use anchor_lang::prelude::*;
use ruint::aliases::U256;
#[derive(AnchorSerialize, AnchorDeserialize, Debug, Clone)]
pub struct BinLiquidityReduction {
    pub bin_id: i32,
    pub bps_to_remove: u16,
}

pub fn calculate_shares_to_remove(bps: u16, bin_id: i32, position: &PositionV2) -> Result<u128> {
    let share_in_bin = U256::from(position.get_liquidity_share_in_bin(bin_id)?);

    let share_to_remove: u128 = U256::from(bps)
        .safe_mul(share_in_bin)?
        .safe_div(U256::from(BASIS_POINT_MAX))?
        .try_into()
        .map_err(|_| LBError::TypeCastFailed)?;
    Ok(share_to_remove)
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
    bin_liquidity_reduction: Vec<BinLiquidityReduction>,
) -> Result<()> {
    Ok(())
}
