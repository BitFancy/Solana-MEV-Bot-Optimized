use crate::ModifyLiquidityOneSide;
use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize, Debug, Clone)]
pub struct AddLiquiditySingleSidePreciseParameter {
    pub bins: Vec<CompressedBinDepositAmount>,
    pub decompress_multiplier: u64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Debug, Clone)]
pub struct CompressedBinDepositAmount {
    pub bin_id: i32,
    pub amount: u32,
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidityOneSide<'info>>,
    parameter: AddLiquiditySingleSidePreciseParameter,
) -> Result<()> {
    Ok(())
}
