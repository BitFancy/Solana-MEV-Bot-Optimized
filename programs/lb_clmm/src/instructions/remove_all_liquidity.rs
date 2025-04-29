use super::add_liquidity::ModifyLiquidity;

use anchor_lang::prelude::*;

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
) -> Result<()> {
    Ok(())
}
