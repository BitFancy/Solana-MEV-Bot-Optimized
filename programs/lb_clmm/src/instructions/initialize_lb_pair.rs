use crate::constants::DEFAULT_OBSERVATION_LENGTH;
use crate::errors::LBError;
use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::lb_pair::LbPair;
use crate::state::oracle::Oracle;
use crate::state::preset_parameters::PresetParameter;
use crate::utils::seeds::BIN_ARRAY_BITMAP_SEED;
use crate::utils::seeds::ORACLE;
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};
use std::cmp::{max, min};

#[event_cpi]
#[derive(Accounts)]
#[instruction(active_id: i32, bin_step: u16)]
pub struct InitializeLbPair<'info> {
    #[account(
        init,
        seeds = [
            min(token_mint_x.key(), token_mint_y.key()).as_ref(),
            max(token_mint_x.key(), token_mint_y.key()).as_ref(),
            &bin_step.to_le_bytes(),
            &preset_parameter.base_factor.to_le_bytes()
        ],
        bump,
        payer = funder,
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
        payer = funder,
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
        payer = funder,
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
        payer = funder,
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
        payer = funder,
        space = Oracle::space(DEFAULT_OBSERVATION_LENGTH)
    )]
    pub oracle: AccountLoader<'info, Oracle>,

    #[account(
        constraint = bin_step == preset_parameter.bin_step @ LBError::NonPresetBinStep,
    )]
    pub preset_parameter: Account<'info, PresetParameter>,

    #[account(mut)]
    pub funder: Signer<'info>,

    // #[account(address = Token2022::id())]
    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handle(ctx: Context<InitializeLbPair>, active_id: i32, bin_step: u16) -> Result<()> {
    Ok(())
}
