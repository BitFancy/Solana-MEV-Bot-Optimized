use crate::state::{bin::BinArray, lb_pair::LbPair};
use crate::utils::seeds::BIN_ARRAY;
use anchor_lang::prelude::*;

#[derive(Accounts)]
#[instruction(index: i64)]
pub struct InitializeBinArray<'info> {
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        init,
        payer = funder,
        seeds = [
            BIN_ARRAY,
            lb_pair.key().as_ref(),
            &index.to_le_bytes()
        ],
        bump,
        space = 8 + BinArray::INIT_SPACE
    )]
    pub bin_array: AccountLoader<'info, BinArray>,

    #[account(mut)]
    pub funder: Signer<'info>,

    pub system_program: Program<'info, System>,
}

pub fn handle(ctx: Context<InitializeBinArray>, index: i64) -> Result<()> {
    Ok(())
}
