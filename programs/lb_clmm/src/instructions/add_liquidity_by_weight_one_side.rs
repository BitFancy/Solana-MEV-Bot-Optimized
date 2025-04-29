use crate::authorize_modify_position;
use crate::constants::MAX_BIN_PER_POSITION;
use crate::errors::LBError;
use crate::math::weight_to_amounts::to_amount_ask_side;
use crate::math::weight_to_amounts::to_amount_bid_side;
use crate::state::bin_array_bitmap_extension::BinArrayBitmapExtension;
use crate::state::position::PositionV2;
use crate::state::{bin::BinArray, lb_pair::LbPair};
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

use super::add_liquidity_by_weight::BinLiquidityDistributionByWeight;

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug)]
pub struct LiquidityOneSideParameter {
    /// Amount of X token or Y token to deposit
    pub amount: u64,
    /// Active bin that integrator observe off-chain
    pub active_id: i32,
    /// max active bin slippage allowed
    pub max_active_bin_slippage: i32,
    /// Liquidity distribution to each bins
    pub bin_liquidity_dist: Vec<BinLiquidityDistributionByWeight>,
}

impl LiquidityOneSideParameter {
    fn bin_count(&self) -> u32 {
        self.bin_liquidity_dist.len() as u32
    }

    fn validate<'a, 'info>(&'a self, active_id: i32) -> Result<()> {
        require!(self.amount != 0, LBError::InvalidInput);

        let bin_count = self.bin_count();
        require!(bin_count > 0, LBError::InvalidInput);

        require!(
            bin_count <= MAX_BIN_PER_POSITION as u32,
            LBError::InvalidInput
        );

        let bin_shift = if active_id > self.active_id {
            active_id - self.active_id
        } else {
            self.active_id - active_id
        };

        require!(
            bin_shift <= self.max_active_bin_slippage.into(),
            LBError::ExceededBinSlippageTolerance
        );

        // bin dist must be in consecutive order and weight is non-zero
        for (i, val) in self.bin_liquidity_dist.iter().enumerate() {
            require!(val.weight != 0, LBError::InvalidInput);
            // bin id must in right order
            if i != 0 {
                require!(
                    val.bin_id > self.bin_liquidity_dist[i - 1].bin_id,
                    LBError::InvalidInput
                );
            }
        }
        Ok(())
    }

    // require bin id to be sorted before doing this
    fn to_amounts_into_bin<'a, 'info>(
        &'a self,
        active_id: i32,
        bin_step: u16,
        deposit_for_y: bool,
    ) -> Result<Vec<(i32, u64)>> {
        if deposit_for_y {
            to_amount_bid_side(
                active_id,
                self.amount,
                &self
                    .bin_liquidity_dist
                    .iter()
                    .map(|x| (x.bin_id, x.weight))
                    .collect::<Vec<(i32, u16)>>(),
            )
        } else {
            to_amount_ask_side(
                active_id,
                self.amount,
                bin_step,
                &self
                    .bin_liquidity_dist
                    .iter()
                    .map(|x| (x.bin_id, x.weight))
                    .collect::<Vec<(i32, u16)>>(),
            )
        }
    }
}

#[event_cpi]
#[derive(Accounts)]
pub struct ModifyLiquidityOneSide<'info> {
    #[account(
        mut,
        has_one = lb_pair,
        constraint = authorize_modify_position(&position, sender.key())?
    )]
    pub position: AccountLoader<'info, PositionV2>,

    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(
        mut,
        has_one = lb_pair,
    )]
    pub bin_array_bitmap_extension: Option<AccountLoader<'info, BinArrayBitmapExtension>>,

    #[account(mut)]
    pub user_token: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(mut)]
    pub reserve: Box<InterfaceAccount<'info, TokenAccount>>,

    pub token_mint: Box<InterfaceAccount<'info, Mint>>,

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
    pub token_program: Interface<'info, TokenInterface>,
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: &Context<'a, 'b, 'c, 'info, ModifyLiquidityOneSide<'info>>,
    liquidity_parameter: &LiquidityOneSideParameter,
) -> Result<()> {
    Ok(())
}
