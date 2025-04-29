use crate::assert_eq_admin;
use crate::constants::NUM_REWARDS;
use crate::errors::LBError;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[event_cpi]
#[derive(Accounts)]
#[instruction(reward_index: u64)]
pub struct UpdateRewardFunder<'info> {
    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin)]
    pub admin: Signer<'info>,
}

pub fn handle(ctx: Context<UpdateRewardFunder>, index: u64, new_funder: Pubkey) -> Result<()> {
    Ok(())
}
