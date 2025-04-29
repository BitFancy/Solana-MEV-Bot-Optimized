use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::bin::BinArray;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[event_cpi]
#[derive(Accounts)]
#[instruction(reward_index: u64)]
pub struct UpdateRewardDuration<'info> {
    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin,
    )]
    pub admin: Signer<'info>,

    #[account(
        mut,
        has_one = lb_pair
    )]
    pub bin_array: AccountLoader<'info, BinArray>,
}

pub fn handle(
    ctx: Context<UpdateRewardDuration>,
    index: u64,
    new_reward_duration: u64,
) -> Result<()> {
    Ok(())
}
