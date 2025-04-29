use crate::authorize_claim_fee_position;
use crate::state::{bin::BinArray, lb_pair::LbPair, position::PositionV2};
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};
#[event_cpi]
#[derive(Accounts)]
pub struct ClaimFee<'info> {
    #[account(
        mut,
        has_one = reserve_x,
        has_one = reserve_y,
        has_one = token_x_mint,
        has_one = token_y_mint,
    )]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        mut,
        has_one = lb_pair,
        constraint = authorize_claim_fee_position(&position, sender.key())?
    )]
    pub position: AccountLoader<'info, PositionV2>,

    #[account(
        mut,
        has_one = lb_pair
    )]
    pub bin_array_lower: AccountLoader<'info, BinArray>,
    #[account(
        mut,
        has_one = lb_pair
    )]
    pub bin_array_upper: AccountLoader<'info, BinArray>,

    pub sender: Signer<'info>,

    #[account(mut)]
    pub reserve_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub reserve_y: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(mut)]
    pub user_token_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub user_token_y: Box<InterfaceAccount<'info, TokenAccount>>,

    pub token_x_mint: Box<InterfaceAccount<'info, Mint>>,
    pub token_y_mint: Box<InterfaceAccount<'info, Mint>>,

    pub token_program: Interface<'info, TokenInterface>,
}

pub fn handle(ctx: Context<ClaimFee>) -> Result<()> {
    Ok(())
}
