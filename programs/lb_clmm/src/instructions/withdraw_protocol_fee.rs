use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

#[derive(Accounts)]
pub struct WithdrawProtocolFee<'info> {
    #[account(
        mut,
        has_one = reserve_x,
        has_one = reserve_y,
        has_one = token_x_mint,
        has_one = token_y_mint,
    )]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(mut)]
    pub reserve_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub reserve_y: Box<InterfaceAccount<'info, TokenAccount>>,

    pub token_x_mint: Box<InterfaceAccount<'info, Mint>>,
    pub token_y_mint: Box<InterfaceAccount<'info, Mint>>,

    #[account(mut)]
    pub receiver_token_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub receiver_token_y: Box<InterfaceAccount<'info, TokenAccount>>,

    pub fee_owner: Signer<'info>,

    pub token_x_program: Interface<'info, TokenInterface>,
    pub token_y_program: Interface<'info, TokenInterface>,
}

pub fn handle(ctx: Context<WithdrawProtocolFee>, amount_x: u64, amount_y: u64) -> Result<()> {
    Ok(())
}
