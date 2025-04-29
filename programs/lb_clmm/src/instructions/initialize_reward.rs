use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

#[event_cpi]
#[derive(Accounts)]
#[instruction(reward_index: u64)]
pub struct InitializeReward<'info> {
    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        init,
        seeds = [
            lb_pair.key().as_ref(),
            reward_index.to_le_bytes().as_ref()
        ],
        bump,
        payer = admin,
        token::mint = reward_mint,
        token::authority = lb_pair,
    )]
    pub reward_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    pub reward_mint: Box<InterfaceAccount<'info, Mint>>,

    #[account(
        mut,
        constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin,
    )]
    pub admin: Signer<'info>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handle(
    ctx: Context<InitializeReward>,
    index: u64,
    reward_duration: u64,
    funder: Pubkey,
) -> Result<()> {
    Ok(())
}
