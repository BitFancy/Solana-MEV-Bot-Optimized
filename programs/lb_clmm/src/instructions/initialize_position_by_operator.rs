use crate::state::lb_pair::LbPair;
use crate::state::position::PositionV2;
use crate::utils::seeds;
use anchor_lang::prelude::*;

#[event_cpi]
#[derive(Accounts)]
#[instruction(lower_bin_id: i32, width: i32)]
pub struct InitializePositionByOperator<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    pub base: Signer<'info>,
    #[account(
        init,        
        seeds = [
            seeds::POSITION.as_ref(),
            lb_pair.key().as_ref(),
            base.key().as_ref(),
            lower_bin_id.to_le_bytes().as_ref(),
            width.to_le_bytes().as_ref(),
        ],
        bump,
        payer = payer,
        space = 8 + PositionV2::INIT_SPACE,
    )]
    pub position: AccountLoader<'info, PositionV2>,

    pub lb_pair: AccountLoader<'info, LbPair>,

    /// operator
    pub operator: Signer<'info>,

    pub system_program: Program<'info, System>,

    pub rent: Sysvar<'info, Rent>,
}

/// There is scenario that operator create and deposit position with non-valid owner
/// Then fund will be lost forever, so only whitelisted operators are able to perform this action
pub fn handle(
    ctx: Context<InitializePositionByOperator>,
    lower_bin_id: i32,
    width: i32,
    owner: Pubkey,
    fee_owner: Pubkey,
) -> Result<()> {
    Ok(())
}
