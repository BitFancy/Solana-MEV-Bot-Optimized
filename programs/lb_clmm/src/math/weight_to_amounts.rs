use crate::errors::LBError;
use crate::math::price_math::get_price_from_id;
use crate::math::safe_math::SafeMath;
use crate::math::u64x64_math::SCALE_OFFSET;
use crate::math::utils_math::safe_mul_div_cast_from_u256_to_u64;
use crate::math::utils_math::safe_mul_div_cast_from_u64_to_u64;
use anchor_lang::prelude::*;
use ruint::aliases::U256;

pub fn to_amount_bid_side(
    active_id: i32,
    amount: u64,
    weights: &[(i32, u16)],
) -> Result<Vec<(i32, u64)>> {
    // get sum of weight
    let mut total_weight = 0u64;
    for &(bin_id, weight) in weights.iter() {
        // skip all ask side
        if bin_id > active_id {
            // break because bin_id is in ascending order
            break;
        }
        total_weight = total_weight.safe_add(weight.into())?;
    }
    if total_weight == 0 {
        return Err(LBError::InvalidInput.into());
    }
    let mut amounts = vec![];
    for &(bin_id, weight) in weights.iter() {
        // skip all ask side
        if bin_id > active_id {
            amounts.push((bin_id, 0));
        } else {
            amounts.push((
                bin_id,
                safe_mul_div_cast_from_u64_to_u64(weight.into(), amount, total_weight)?,
            ));
        }
    }
    Ok(amounts)
}

pub fn to_amount_ask_side(
    active_id: i32,
    amount: u64,
    bin_step: u16,
    weights: &[(i32, u16)],
) -> Result<Vec<(i32, u64)>> {
    // get sum of weight
    let mut total_weight = U256::ZERO;
    let mut weight_per_prices = vec![U256::ZERO; weights.len()];
    for (i, &(bin_id, weight)) in weights.iter().enumerate() {
        // skip all bid side
        if bin_id < active_id {
            continue;
        }
        let weight_per_price = U256::from(weight)
            .safe_shl((SCALE_OFFSET * 2).into())?
            .safe_div(U256::from(get_price_from_id(bin_id, bin_step)?))?;
        weight_per_prices[i] = weight_per_price;
        total_weight = total_weight.safe_add(weight_per_price)?;
    }

    if total_weight == U256::ZERO {
        return Err(LBError::InvalidInput.into());
    }

    let mut amounts = vec![];
    for (i, &(bin_id, _weight)) in weights.iter().enumerate() {
        // skip all bid side
        if bin_id < active_id {
            amounts.push((bin_id, 0));
        } else {
            amounts.push((
                bin_id,
                safe_mul_div_cast_from_u256_to_u64(amount, weight_per_prices[i], total_weight)?,
            ));
        }
    }
    Ok(amounts)
}

fn get_active_bin_index(active_id: i32, weights: &[(i32, u16)]) -> Option<usize> {
    for (i, &(bin_id, _weight)) in weights.iter().enumerate() {
        if bin_id == active_id {
            return Some(i);
        }
        // bin_id is sorted, so no need to check if bin cross
        if bin_id > active_id {
            break;
        }
    }
    return None;
}

pub fn to_amount_both_side(
    active_id: i32,
    bin_step: u16,
    amount_x: u64, // amount_x in active bin
    amount_y: u64, // amount_y in active bin
    total_amount_x: u64,
    total_amount_y: u64,
    weights: &[(i32, u16)],
) -> Result<Vec<(i32, u64, u64)>> {
    match get_active_bin_index(active_id, weights) {
        Some(index) => {
            let (active_bin_id, active_weight) = weights[index];
            let p0 = U256::from(get_price_from_id(active_bin_id, bin_step)?);

            let (wx0, wy0) = if amount_x == 0 && amount_y == 0 {
                // equal ratio if both amount_x and amount_y is zero
                let wx0 = U256::from(active_weight)
                    .safe_shl((SCALE_OFFSET * 2).into())?
                    .safe_div(p0.safe_mul(U256::from(2))?)?;

                let wy0 = U256::from(active_weight)
                    .safe_shl(SCALE_OFFSET.into())?
                    .safe_div(U256::from(2))?;
                (wx0, wy0)
            } else {
                let wx0 = if amount_x == 0 {
                    U256::ZERO
                } else {
                    U256::from(active_weight)
                        .safe_shl((SCALE_OFFSET * 2).into())?
                        .safe_div(
                            p0.safe_add(
                                U256::from(amount_y)
                                    .safe_shl(SCALE_OFFSET.into())?
                                    .safe_div(U256::from(amount_x))?,
                            )?,
                        )?
                };
                let wy0 = if amount_y == 0 {
                    U256::ZERO
                } else {
                    U256::from(active_weight)
                        .safe_shl((SCALE_OFFSET * 2).into())?
                        .safe_div(
                            U256::from(1).safe_shl(SCALE_OFFSET.into())?.safe_add(
                                p0.safe_mul(U256::from(amount_x))?
                                    .safe_div(U256::from(amount_y))?,
                            )?,
                        )?
                };
                (wx0, wy0)
            };

            let mut total_weight_x = wx0;
            let mut total_weight_y = wy0;
            let mut weight_per_prices = vec![U256::ZERO; weights.len()];
            for (i, &(bin_id, weight)) in weights.iter().enumerate() {
                if bin_id < active_id {
                    total_weight_y = total_weight_y
                        .safe_add(U256::from(weight).safe_shl(SCALE_OFFSET.into())?)?;
                    continue;
                }
                if bin_id > active_id {
                    let weight_per_price = U256::from(weight)
                        .safe_shl((SCALE_OFFSET * 2).into())?
                        .safe_div(U256::from(get_price_from_id(bin_id, bin_step)?))?;
                    weight_per_prices[i] = weight_per_price;
                    total_weight_x = total_weight_x.safe_add(weight_per_price)?;
                    continue;
                }
            }
            // find k
            let ky = U256::from(total_amount_y)
                .safe_shl((SCALE_OFFSET * 2).into())?
                .safe_div(total_weight_y)?;

            let kx = U256::from(total_amount_x)
                .safe_shl((SCALE_OFFSET * 2).into())?
                .safe_div(total_weight_x)?;
            let k = kx.min(ky);
            let mut amounts = vec![];
            for (i, &(bin_id, weight)) in weights.iter().enumerate() {
                if bin_id < active_id {
                    let (amount_y_in_bin, _) = k
                        .safe_mul(U256::from(weight))?
                        .overflowing_shr(SCALE_OFFSET.into());
                    amounts.push((
                        bin_id,
                        0,
                        u64::try_from(amount_y_in_bin).map_err(|_| LBError::TypeCastFailed)?,
                    ));
                    continue;
                }
                if bin_id > active_id {
                    let (amount_x_in_bin, _) = k
                        .safe_mul(weight_per_prices[i])?
                        .overflowing_shr((SCALE_OFFSET * 2).into());

                    amounts.push((
                        bin_id,
                        u64::try_from(amount_x_in_bin).map_err(|_| LBError::TypeCastFailed)?,
                        0,
                    ));
                    continue;
                }
                // else we are in active id
                let (amount_x_in_bin, _) =
                    k.safe_mul(wx0)?.overflowing_shr((SCALE_OFFSET * 2).into());
                let (amount_y_in_bin, _) =
                    k.safe_mul(wy0)?.overflowing_shr((SCALE_OFFSET * 2).into());

                amounts.push((
                    bin_id,
                    u64::try_from(amount_x_in_bin).map_err(|_| LBError::TypeCastFailed)?,
                    u64::try_from(amount_y_in_bin).map_err(|_| LBError::TypeCastFailed)?,
                ));
            }
            return Ok(amounts);
        }
        None => {
            let mut total_weight_x = U256::ZERO;
            let mut total_weight_y = U256::ZERO;
            let mut weight_per_prices = vec![U256::ZERO; weights.len()];

            for (i, &(bin_id, weight)) in weights.iter().enumerate() {
                if bin_id < active_id {
                    total_weight_y = total_weight_y
                        .safe_add(U256::from(weight).safe_shl(SCALE_OFFSET.into())?)?;
                    continue;
                }
                if bin_id > active_id {
                    let weight_per_price = U256::from(weight)
                        .safe_shl((SCALE_OFFSET * 2).into())?
                        .safe_div(U256::from(get_price_from_id(bin_id, bin_step)?))?;
                    weight_per_prices[i] = weight_per_price;
                    total_weight_x = total_weight_x.safe_add(weight_per_price)?;
                }
            }
            // find k
            let ky = U256::from(total_amount_y)
                .safe_shl((SCALE_OFFSET * 2).into())?
                .safe_div(total_weight_y)?;

            let kx = U256::from(total_amount_x)
                .safe_shl((SCALE_OFFSET * 2).into())?
                .safe_div(total_weight_x)?;
            let k = kx.min(ky);

            let mut amounts = vec![];
            for (i, &(bin_id, weight)) in weights.iter().enumerate() {
                if bin_id < active_id {
                    let (amount_y_in_bin, _) = k
                        .safe_mul(U256::from(weight))?
                        .overflowing_shr(SCALE_OFFSET.into());
                    amounts.push((
                        bin_id,
                        0,
                        u64::try_from(amount_y_in_bin).map_err(|_| LBError::TypeCastFailed)?,
                    ));
                    continue;
                }
                if bin_id > active_id {
                    let (amount_x_in_bin, _) = k
                        .safe_mul(weight_per_prices[i])?
                        .overflowing_shr((SCALE_OFFSET * 2).into());

                    amounts.push((
                        bin_id,
                        u64::try_from(amount_x_in_bin).map_err(|_| LBError::TypeCastFailed)?,
                        0,
                    ));
                }
            }
            return Ok(amounts);
        }
    }
}
