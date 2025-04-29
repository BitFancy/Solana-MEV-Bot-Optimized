use crate::constants::MAX_BIN_PER_POSITION;
use crate::errors::LBError;
use crate::math::weight_to_amounts::{to_amount_ask_side, to_amount_bid_side, to_amount_both_side};
use crate::ModifyLiquidity;
use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug, Default)]
pub struct BinLiquidityDistributionByWeight {
    /// Define the bin ID wish to deposit to.
    pub bin_id: i32,
    /// weight of liquidity distributed for this bin id
    pub weight: u16,
}

#[derive(AnchorSerialize, AnchorDeserialize, Eq, PartialEq, Clone, Debug)]
pub struct LiquidityParameterByWeight {
    /// Amount of X token to deposit
    pub amount_x: u64,
    /// Amount of Y token to deposit
    pub amount_y: u64,
    /// Active bin that integrator observe off-chain
    pub active_id: i32,
    /// max active bin slippage allowed
    pub max_active_bin_slippage: i32,
    /// Liquidity distribution to each bins
    pub bin_liquidity_dist: Vec<BinLiquidityDistributionByWeight>,
}

impl LiquidityParameterByWeight {
    fn bin_count(&self) -> u32 {
        self.bin_liquidity_dist.len() as u32
    }

    pub fn validate<'a, 'info>(&'a self, active_id: i32) -> Result<()> {
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
        let first_bin_id = self.bin_liquidity_dist[0].bin_id;
        let last_bin_id = self.bin_liquidity_dist[self.bin_liquidity_dist.len() - 1].bin_id;

        if first_bin_id > active_id {
            require!(self.amount_x != 0, LBError::InvalidInput);
        }
        if last_bin_id < active_id {
            require!(self.amount_y != 0, LBError::InvalidInput);
        }

        Ok(())
    }

    // require bin id to be sorted before doing this
    pub fn to_amounts_into_bin<'a, 'info>(
        &'a self,
        active_id: i32,
        bin_step: u16,
        amount_x_in_active_bin: u64, // amount x in active bin
        amount_y_in_active_bin: u64, // amount y in active bin
    ) -> Result<Vec<(i32, u64, u64)>> {
        // only bid side
        if active_id > self.bin_liquidity_dist[self.bin_liquidity_dist.len() - 1].bin_id {
            let amounts = to_amount_bid_side(
                active_id,
                self.amount_y,
                &self
                    .bin_liquidity_dist
                    .iter()
                    .map(|x| (x.bin_id, x.weight))
                    .collect::<Vec<(i32, u16)>>(),
            )?;

            let amounts = amounts
                .iter()
                .map(|x| (x.0, 0, x.1))
                .collect::<Vec<(i32, u64, u64)>>();

            return Ok(amounts);
        }
        // only ask side
        if active_id < self.bin_liquidity_dist[0].bin_id {
            let amounts = to_amount_ask_side(
                active_id,
                self.amount_x,
                bin_step,
                &self
                    .bin_liquidity_dist
                    .iter()
                    .map(|x| (x.bin_id, x.weight))
                    .collect::<Vec<(i32, u16)>>(),
            )?;

            let amounts = amounts
                .iter()
                .map(|x| (x.0, x.1, 0))
                .collect::<Vec<(i32, u64, u64)>>();

            return Ok(amounts);
        }

        to_amount_both_side(
            active_id,
            bin_step,
            amount_x_in_active_bin,
            amount_y_in_active_bin,
            self.amount_x,
            self.amount_y,
            &self
                .bin_liquidity_dist
                .iter()
                .map(|x| (x.bin_id, x.weight))
                .collect::<Vec<(i32, u16)>>(),
        )
    }
}

pub fn handle<'a, 'b, 'c, 'info>(
    ctx: &Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
    liquidity_parameter: &LiquidityParameterByWeight,
) -> Result<()> {
    Ok(())
}
