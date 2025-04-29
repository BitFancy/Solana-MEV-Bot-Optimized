use crate::authorize_modify_position;
use crate::state::{bin::BinArray, lb_pair::LbPair, position::PositionV2};
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct UpdateFeesAndRewards<'info> {
    #[account(
        mut,
        has_one = lb_pair,
        constraint = authorize_modify_position(&position, owner.key())?
    )]
    pub position: AccountLoader<'info, PositionV2>,

    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        mut,
        has_one = lb_pair
    )]
    pub bin_array_lower: AccountLoader<'info, BinArray>,
    #[account(
        mut,
        has_one = lb_pair
    )]
    pub bin_array_upper: AccountLoader<'info, BinArray>,

    pub owner: Signer<'info>,
}

pub fn handle(ctx: Context<UpdateFeesAndRewards>) -> Result<()> {
    Ok(())
}
