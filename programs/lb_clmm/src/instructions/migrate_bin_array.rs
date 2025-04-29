use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct MigrateBinArray<'info> {
    pub lb_pair: AccountLoader<'info, LbPair>,
}

pub fn handle(ctx: Context<MigrateBinArray>) -> Result<()> {
    Ok(())
}
