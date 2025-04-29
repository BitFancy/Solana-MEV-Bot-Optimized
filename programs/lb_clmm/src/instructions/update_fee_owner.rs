use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct UpdateFeeOwner<'info> {
    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    /// CHECK: New fee owner doesn't need to be deserialized.
    #[account(
        constraint = lb_pair.load()?.fee_owner != new_fee_owner.key() @ LBError::IdenticalFeeOwner,
    )]
    pub new_fee_owner: UncheckedAccount<'info>,

    #[account(
        constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin,
    )]
    pub admin: Signer<'info>,
}

pub fn handle(ctx: Context<UpdateFeeOwner>) -> Result<()> {
    Ok(())
}
