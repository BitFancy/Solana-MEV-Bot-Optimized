use crate::authorize_modify_position;
use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::position::PositionV2;
use crate::state::{bin::BinArray, lb_pair::LbPair};
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

pub struct CompositeDepositInfo {
    pub liquidity_share: u128,
    pub protocol_token_x_fee_amount: u64,
    pub protocol_token_y_fee_amount: u64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug)]
pub struct BinLiquidityDistribution {
    /// Define the bin ID wish to deposit to.
    pub bin_id: i32,
    /// DistributionX (or distributionY) is the percentages of amountX (or amountY) you want to add to each bin.
    pub distribution_x: u16,
    /// DistributionX (or distributionY) is the percentages of amountX (or amountY) you want to add to each bin.
    pub distribution_y: u16,
}

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug)]
pub struct LiquidityParameter {
    /// Amount of X token to deposit
    pub amount_x: u64,
    /// Amount of Y token to deposit
    pub amount_y: u64,
    /// Liquidity distribution to each bins
    pub bin_liquidity_dist: Vec<BinLiquidityDistribution>,
}

#[event_cpi]
#[derive(Accounts)]
pub struct ModifyLiquidity<'info> {
    #[account(
        mut,
        has_one = lb_pair,
        constraint = authorize_modify_position(&position, sender.key())?
    )]
    pub position: AccountLoader<'info, PositionV2>,

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
    )]
    pub bin_array_bitmap_extension: Option<AccountLoader<'info, BinArrayBitmapExtension>>,

    #[account(
        mut,
        token::mint = token_x_mint
    )]
    pub user_token_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(
        mut,
        token::mint = token_y_mint
    )]
    pub user_token_y: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(mut)]
    pub reserve_x: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub reserve_y: Box<InterfaceAccount<'info, TokenAccount>>,

    pub token_x_mint: Box<InterfaceAccount<'info, Mint>>,
    pub token_y_mint: Box<InterfaceAccount<'info, Mint>>,

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
    pub token_x_program: Interface<'info, TokenInterface>,
    pub token_y_program: Interface<'info, TokenInterface>,
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
    liquidity_parameter: LiquidityParameter,
) -> Result<()> {
    Ok(())
}
