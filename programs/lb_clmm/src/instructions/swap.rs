use crate::errors::LBError;
use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::lb_pair::*;
use crate::state::oracle::Oracle;
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

#[event_cpi]
#[derive(Accounts)]
pub struct Swap<'info> {
    #[account(
        mut,
        has_one = reserve_x,
        has_one = reserve_y,
        has_one = token_x_mint,
        has_one = token_y_mint,
        has_one = oracle,
    )]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        has_one = lb_pair,
    )]
    pub bin_array_bitmap_extension: Option<AccountLoader<'info, BinArrayBitmapExtension>>,

    #[account(mut)]
    pub reserve_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub reserve_y: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(
        mut,
        constraint = user_token_in.mint != user_token_out.mint @ LBError::InvalidTokenMint,
        constraint = user_token_in.mint == token_x_mint.key() || user_token_in.mint == token_y_mint.key() @ LBError::InvalidTokenMint,
    )]
    pub user_token_in: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(
        mut,
        constraint = user_token_out.mint == token_x_mint.key() || user_token_out.mint == token_y_mint.key() @ LBError::InvalidTokenMint,
    )]
    pub user_token_out: Box<InterfaceAccount<'info, TokenAccount>>,

    pub token_x_mint: Box<InterfaceAccount<'info, Mint>>,
    pub token_y_mint: Box<InterfaceAccount<'info, Mint>>,

    #[account(mut)]
    pub oracle: AccountLoader<'info, Oracle>,

    #[account(mut)]
    pub host_fee_in: Option<Box<InterfaceAccount<'info, TokenAccount>>>,

    pub user: Signer<'info>,
    pub token_x_program: Interface<'info, TokenInterface>,
    pub token_y_program: Interface<'info, TokenInterface>,
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, Swap<'info>>,
    amount_in: u64,
    min_amount_out: u64,
) -> Result<()> {
    Ok(())
}
