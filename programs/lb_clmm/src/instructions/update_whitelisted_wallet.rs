use crate::{errors::LBError, state::lb_pair::LbPair};
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct UpdateWhitelistWallet<'info> {
    #[account(
        mut,
        has_one = creator
    )]
    pub lb_pair: AccountLoader<'info, LbPair>,

    pub creator: Signer<'info>,
}

pub fn handle(ctx: Context<UpdateWhitelistWallet>, idx: u8, wallet: Pubkey) -> Result<()> {
    Ok(())
}
