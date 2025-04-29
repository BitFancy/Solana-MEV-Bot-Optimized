use super::{
    safe_math::SafeMath,
    u128x128_math::{mul_div, mul_shr, shl_div, Rounding},
    u64x64_math::pow,
};
use crate::errors::LBError;
use anchor_lang::prelude::Result;
use num_traits::cast::FromPrimitive;
use ruint::{aliases::U256, Uint};

#[inline]
pub fn safe_pow_cast<T: FromPrimitive>(base: u128, exp: i32) -> Result<T> {
    T::from_u128(pow(base, exp).ok_or_else(|| LBError::MathOverflow)?)
        .ok_or_else(|| LBError::TypeCastFailed.into())
}

#[inline]
pub fn safe_mul_div_cast<T: FromPrimitive>(
    x: u128,
    y: u128,
    denominator: u128,
    rounding: Rounding,
) -> Result<T> {
    T::from_u128(mul_div(x, y, denominator, rounding).ok_or_else(|| LBError::MathOverflow)?)
        .ok_or_else(|| LBError::TypeCastFailed.into())
}

#[inline]
pub fn safe_mul_div_cast_from_u64_to_u64(x: u64, y: u64, denominator: u64) -> Result<u64> {
    let x = u128::from(x);
    let y = u128::from(y);
    let denominator = u128::from(denominator);
    let result = u64::try_from(x.safe_mul(y)?.safe_div(denominator)?)
        .map_err(|_| LBError::TypeCastFailed)?;
    Ok(result)
}

#[inline]
pub fn safe_mul_div_cast_from_u256_to_u64(x: u64, y: U256, denominator: U256) -> Result<u64> {
    let x = U256::from(x);
    // let denominator = U256::from(denominator);
    let result = u64::try_from(x.safe_mul(y)?.safe_div(denominator)?)
        .map_err(|_| LBError::TypeCastFailed)?;
    Ok(result)
}

#[inline]
pub fn safe_mul_shr_cast<T: FromPrimitive>(
    x: u128,
    y: u128,
    offset: u8,
    rounding: Rounding,
) -> Result<T> {
    T::from_u128(mul_shr(x, y, offset, rounding).ok_or_else(|| LBError::MathOverflow)?)
        .ok_or_else(|| LBError::TypeCastFailed.into())
}

#[inline]
pub fn safe_shl_div_cast<T: FromPrimitive>(
    x: u128,
    y: u128,
    offset: u8,
    rounding: Rounding,
) -> Result<T> {
    T::from_u128(shl_div(x, y, offset, rounding).ok_or_else(|| LBError::MathOverflow)?)
        .ok_or_else(|| LBError::TypeCastFailed.into())
}

pub const fn one<const BITS: usize, const LIMBS: usize>() -> Uint<BITS, LIMBS> {
    let mut words = [0; LIMBS];
    words[0] = 1;
    Uint::from_limbs(words)
}
