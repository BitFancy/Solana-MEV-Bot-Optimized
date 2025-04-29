use crate::state::position::PositionV2;
use anchor_lang::prelude::*;
#[event_cpi]
#[derive(Accounts)]
pub struct UpdatePositionOperator<'info> {
    #[account(mut, has_one = owner)]
    pub position: AccountLoader<'info, PositionV2>,
    pub owner: Signer<'info>,
}

pub fn handle(ctx: Context<UpdatePositionOperator>, new_operator: Pubkey) -> Result<()> {
    Ok(())
}
