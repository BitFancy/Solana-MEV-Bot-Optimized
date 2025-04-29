use crate::assert_eq_launch_pool_admin;
use crate::constants::DEFAULT_OBSERVATION_LENGTH;
use crate::errors::LBError;
use crate::events::LbPairCreate;
use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::lb_pair::LbPair;
use crate::state::lb_pair::PairType;
use crate::state::oracle::Oracle;
use crate::state::preset_parameters::PresetParameter;
use crate::utils::seeds::BIN_ARRAY_BITMAP_SEED;
use crate::utils::seeds::ORACLE;
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};
use std::cmp::{max, min};

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct InitPermissionPairIx {
    pub active_id: i32,
    pub bin_step: u16,
    pub base_factor: u16,
    pub min_bin_id: i32,
    pub max_bin_id: i32,
    pub lock_duration_in_slot: u64,
}

#[event_cpi]
#[derive(Accounts)]
#[instruction(ix_data: InitPermissionPairIx)]
pub struct InitializePermissionLbPair<'info> {
    pub base: Signer<'info>,

    #[account(
        init,
        seeds = [
            base.key().as_ref(),
            min(token_mint_x.key(), token_mint_y.key()).as_ref(),
            max(token_mint_x.key(), token_mint_y.key()).as_ref(),
            &ix_data.bin_step.to_le_bytes(),
        ],
        bump,
        payer = admin,
        space = 8 + LbPair::INIT_SPACE
    )]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        init,
        seeds = [
            BIN_ARRAY_BITMAP_SEED,
            lb_pair.key().as_ref(),
        ],
        bump,
        payer = admin,
        space = 8 + BinArrayBitmapExtension::INIT_SPACE
    )]
    pub bin_array_bitmap_extension: Option<AccountLoader<'info, BinArrayBitmapExtension>>,

    #[account(constraint = token_mint_x.key() != token_mint_y.key())]
    pub token_mint_x: Box<InterfaceAccount<'info, Mint>>,
    pub token_mint_y: Box<InterfaceAccount<'info, Mint>>,

    #[account(
        init,
        seeds = [
            lb_pair.key().as_ref(),
            token_mint_x.key().as_ref()
        ],
        bump,
        payer = admin,
        token::mint = token_mint_x,
        token::authority = lb_pair,
    )]
    pub reserve_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(
        init,
        seeds = [
            lb_pair.key().as_ref(),
            token_mint_y.key().as_ref()
        ],
        bump,
        payer = admin,
        token::mint = token_mint_y,
        token::authority = lb_pair,
    )]
    pub reserve_y: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(
        init,
        seeds = [
            ORACLE,
            lb_pair.key().as_ref()
        ],
        bump,
        payer = admin,
        space = Oracle::space(DEFAULT_OBSERVATION_LENGTH)
    )]
    pub oracle: AccountLoader<'info, Oracle>,

    #[account(
        mut,
        constraint = assert_eq_launch_pool_admin(admin.key()) @ LBError::InvalidAdmin,
    )]
    pub admin: Signer<'info>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handle(
    ctx: Context<InitializePermissionLbPair>,
    ix_data: InitPermissionPairIx,
) -> Result<()> {
    Ok(())
}
