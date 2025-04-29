use ruint::aliases::U256;

// Round up, down
#[derive(PartialEq)]
pub enum Rounding {
    Up,
    Down,
}

/// (x * y) / denominator
pub fn mul_div(x: u128, y: u128, denominator: u128, rounding: Rounding) -> Option<u128> {
    if denominator == 0 {
        return None;
    }

    let x = U256::from(x);
    let y = U256::from(y);
    let denominator = U256::from(denominator);

    let prod = x.checked_mul(y)?;

    match rounding {
        Rounding::Up => prod.div_ceil(denominator).try_into().ok(),
        Rounding::Down => {
            let (quotient, _) = prod.div_rem(denominator);
            quotient.try_into().ok()
        }
    }
}

/// (x * y) >> offset
#[inline]
pub fn mul_shr(x: u128, y: u128, offset: u8, rounding: Rounding) -> Option<u128> {
    let denominator = 1u128.checked_shl(offset.into())?;
    mul_div(x, y, denominator, rounding)
}

/// (x << offset) / y
#[inline]
pub fn shl_div(x: u128, y: u128, offset: u8, rounding: Rounding) -> Option<u128> {
    let scale = 1u128.checked_shl(offset.into())?;
    mul_div(x, scale, y, rounding)
}
