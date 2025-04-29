use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct TogglePairStatus<'info> {
    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin)]
    pub admin: Signer<'info>,
}

pub fn handle(ctx: Context<TogglePairStatus>) -> Result<()> {
    Ok(())
}
