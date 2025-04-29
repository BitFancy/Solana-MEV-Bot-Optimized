use anchor_lang::prelude::*;

use crate::state::{
    bin::BinArray,
    lb_pair::LbPair,
    position::{Position, PositionV2},
};

#[event_cpi]
#[derive(Accounts)]
pub struct MigratePosition<'info> {
    #[account(
        init,
        payer = owner,
        space = 8 + PositionV2::INIT_SPACE,
    )]
    pub position_v2: AccountLoader<'info, PositionV2>,

    // TODO do we need to check whether it is pda?
    #[account(
        mut,
        has_one = owner,
        has_one = lb_pair,
        close = rent_receiver
    )]
    pub position_v1: AccountLoader<'info, Position>,

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

    #[account(mut)]
    pub owner: Signer<'info>,

    pub system_program: Program<'info, System>,

    /// CHECK: Account to receive closed account rental SOL
    #[account(mut)]
    pub rent_receiver: UncheckedAccount<'info>,
}

pub fn handle(ctx: Context<MigratePosition>) -> Result<()> {
    Ok(())
}
