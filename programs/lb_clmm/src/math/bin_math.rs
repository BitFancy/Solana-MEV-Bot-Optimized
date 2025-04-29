use crate::errors::LBError;

use super::safe_math::SafeMath;
use super::u64x64_math::SCALE_OFFSET;
use anchor_lang::prelude::Result;
use ruint::aliases::U256;

/// Calculate the amount of liquidity following the constant sum formula `L = price * x + y`
/// Price is in Q64x64
pub fn get_liquidity(x: u64, y: u64, price: u128) -> Result<u128> {
    // Q64x0
    let x: U256 = U256::from(x);

    // Multiplication do not require same Q number format. px is in Q64x64
    let price = U256::from(price);
    let px = price.safe_mul(x)?;

    // When perform add, both must be same Q number format. Therefore << SCALE_OFFSET to make y and px Q64x64
    let y = u128::from(y).safe_shl(SCALE_OFFSET.into())?;
    let y = U256::from(y);
    // Liquidity represented with fractional part
    let liquidity = px.safe_add(U256::from(y))?;
    Ok(liquidity.try_into().map_err(|_| LBError::TypeCastFailed)?)
}
