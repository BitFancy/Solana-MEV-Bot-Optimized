use crate::constants::MAX_BIN_PER_ARRAY;
use crate::errors::LBError;
use crate::state::bin::Bin;
use crate::state::lb_pair::LbPair;
use crate::{math::safe_math::SafeMath, state::bin::BinArray};
use anchor_lang::prelude::*;
use std::cell::{Ref, RefMut};

/// A bin arrays container which make sure that the bin array are in continuous form.
pub struct BinArrayManager<'a, 'info> {
    bin_arrays: &'a mut [RefMut<'info, BinArray>],
}

impl<'a, 'info> BinArrayManager<'a, 'info> {
    pub fn new(bin_arrays: &'a mut [RefMut<'info, BinArray>]) -> Result<Self> {
        Ok(BinArrayManager { bin_arrays })
    }

    pub fn migrate_to_v2(&mut self) -> Result<()> {
        // do it every step
        for bin_array in self.bin_arrays.iter_mut() {
            bin_array.migrate_to_v2()?;
        }
        Ok(())
    }

    pub fn get_zero_liquidity_flags(&self) -> Vec<bool> {
        let mut flags = vec![];
        for bin_array in self.bin_arrays.iter() {
            flags.push(bin_array.is_zero_liquidity());
        }
        flags
    }

    pub fn get_bin_array_index(&self, index: usize) -> Result<i32> {
        let index =
            i32::try_from(self.bin_arrays[index].index).map_err(|_| LBError::MathOverflow)?;
        Ok(index)
    }

    /// Validate whether the lower and upper bin array loaded aligned to lower bin id
    pub fn validate_bin_arrays(&self, lower_bin_id: i32) -> Result<()> {
        require!(self.bin_arrays.len() > 0, LBError::InvalidInput);

        let bin_array_0_index = BinArray::bin_id_to_bin_array_index(lower_bin_id)?;

        require!(
            bin_array_0_index as i64 == self.bin_arrays[0].index,
            LBError::InvalidInput
        );

        for i in 0..self.bin_arrays.len().saturating_sub(1) {
            let current_bin_array = &self.bin_arrays[i];
            let next_bin_array = &self.bin_arrays[i + 1];

            require!(
                current_bin_array.index + 1 == next_bin_array.index,
                LBError::NonContinuousBinArrays
            );
        }

        Ok(())
    }

    pub fn get_lower_upper_bin_id(&self) -> Result<(i32, i32)> {
        let lower_bin_array_idx = self.bin_arrays[0].index as i32;
        let upper_bin_array_idx = self.bin_arrays[self.bin_arrays.len() - 1].index as i32;

        let lower_bin_id = lower_bin_array_idx.safe_mul(MAX_BIN_PER_ARRAY as i32)?;
        let upper_bin_id = upper_bin_array_idx
            .safe_mul(MAX_BIN_PER_ARRAY as i32)?
            .safe_add(MAX_BIN_PER_ARRAY as i32)?
            .safe_sub(1)?;

        Ok((lower_bin_id, upper_bin_id))
    }

    pub fn is_bin_id_within_range(&self, bin_id: i32) -> Result<()> {
        let (lower_bin_id, upper_bin_id) = self.get_lower_upper_bin_id()?;

        require!(
            bin_id >= lower_bin_id && bin_id <= upper_bin_id,
            LBError::InvalidBinArray
        );

        Ok(())
    }

    // Update the rewards for active bin. If the active bin doesn't within the bin arrays, nothing will be updated.
    pub fn update_rewards<'b>(&mut self, lb_pair: &mut RefMut<'b, LbPair>) -> Result<()> {
        let current_timestamp = Clock::get()?.unix_timestamp;

        for bin_array in self.bin_arrays.iter_mut() {
            if bin_array.is_bin_id_within_range(lb_pair.active_id).is_ok() {
                bin_array.update_all_rewards(lb_pair, current_timestamp as u64)?;
                break;
            }
        }

        Ok(())
    }

    pub fn get_continuous_bins(&'a self) -> impl Iterator<Item = &'a Bin> {
        self.bin_arrays
            .iter()
            .map(|ba| ba.bins.iter())
            .flat_map(|bins_iter| bins_iter)
    }

    pub fn get_bin_arrays(&'a mut self) -> &'a mut [RefMut<'info, BinArray>] {
        &mut self.bin_arrays
    }

    pub fn get_bin(&self, bin_id: i32) -> Result<&Bin> {
        let bin_array_idx = BinArray::bin_id_to_bin_array_index(bin_id)?;
        match self
            .bin_arrays
            .iter()
            .find(|ba| ba.index == bin_array_idx as i64)
        {
            Some(bin_array) => bin_array.get_bin(bin_id),
            None => Err(LBError::InvalidBinArray.into()),
        }
    }

    pub fn get_bin_mut(&mut self, bin_id: i32) -> Result<&mut Bin> {
        let bin_array_idx = BinArray::bin_id_to_bin_array_index(bin_id)?;
        match self
            .bin_arrays
            .iter_mut()
            .find(|ba| ba.index == bin_array_idx as i64)
        {
            Some(bin_array) => bin_array.get_bin_mut(bin_id),
            None => Err(LBError::InvalidBinArray.into()),
        }
    }
}

pub struct BinArrayManagerReadOnly<'a, 'info> {
    bin_arrays: &'a [Ref<'info, BinArray>],
}

impl<'a, 'info> BinArrayManagerReadOnly<'a, 'info> {
    pub fn new(bin_arrays: &'a [Ref<'info, BinArray>]) -> Result<Self> {
        Ok(BinArrayManagerReadOnly { bin_arrays })
    }

    pub fn get_bin(&self, bin_id: i32) -> Result<&Bin> {
        let bin_array_idx = BinArray::bin_id_to_bin_array_index(bin_id)?;
        match self
            .bin_arrays
            .iter()
            .find(|ba| ba.index == bin_array_idx as i64)
        {
            Some(bin_array) => bin_array.get_bin(bin_id),
            None => Err(LBError::InvalidBinArray.into()),
        }
    }
}
