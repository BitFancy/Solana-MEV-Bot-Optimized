use anchor_lang::prelude::*;

use crate::state::{lb_pair::LbPair, position::PositionV2};

#[event_cpi]
#[derive(Accounts)]
pub struct InitializePosition<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account(
        init,
        payer = payer,
        space = 8 + PositionV2::INIT_SPACE,
    )]
    pub position: AccountLoader<'info, PositionV2>,

    pub lb_pair: AccountLoader<'info, LbPair>,

    pub owner: Signer<'info>,

    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handle(ctx: Context<InitializePosition>, lower_bin_id: i32, width: i32) -> Result<()> {
    Ok(())
}
