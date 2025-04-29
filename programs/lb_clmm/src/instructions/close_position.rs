use anchor_lang::prelude::*;

use crate::authorize_modify_position;
use crate::state::{bin::BinArray, lb_pair::LbPair, position::PositionV2};

#[event_cpi]
#[derive(Accounts)]
pub struct ClosePosition<'info> {
    #[account(
        mut,
        has_one = lb_pair,
        constraint = authorize_modify_position(&position, sender.key())?,
        close = rent_receiver
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

    pub sender: Signer<'info>,

    /// CHECK: Account to receive closed account rental SOL
    #[account(mut)]
    pub rent_receiver: UncheckedAccount<'info>,
}

pub fn handle(ctx: Context<ClosePosition>) -> Result<()> {
    Ok(())
}
