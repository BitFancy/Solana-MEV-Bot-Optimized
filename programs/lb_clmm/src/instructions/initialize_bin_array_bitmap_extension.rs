use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::lb_pair::LbPair;
use crate::utils::seeds::BIN_ARRAY_BITMAP_SEED;
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct InitializeBinArrayBitmapExtension<'info> {
    pub lb_pair: AccountLoader<'info, LbPair>,
    /// Initialize an account to store if a bin array is initialized.
    #[account(
        init,
        seeds = [
            BIN_ARRAY_BITMAP_SEED,
            lb_pair.key().as_ref(),
        ],
        bump,
        payer = funder,
        space = 8 + BinArrayBitmapExtension::INIT_SPACE
    )]
    pub bin_array_bitmap_extension: AccountLoader<'info, BinArrayBitmapExtension>,
    #[account(mut)]
    pub funder: Signer<'info>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handle(ctx: Context<InitializeBinArrayBitmapExtension>) -> Result<()> {
    Ok(())
}
