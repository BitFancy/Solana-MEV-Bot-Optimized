use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::MathematicalOps;
use rust_decimal_macros::dec;

pub fn from_x64_orca_wp(num: u128, decimals_0: f64, decimals_1: f64) -> Decimal {
    println!("numX64: {:?}", num);
    
    let num_dec: Decimal = Decimal::from_u128(num).unwrap();
    let mul_x64: f64 = 2f64.powf(-64.0);
    let from_x64: Decimal = num_dec.checked_mul(Decimal::from_f64_retain(mul_x64).unwrap()).unwrap();
    let price: Option<Decimal> = from_x64.checked_powd(dec!(2)).unwrap().checked_mul(Decimal::from_f64_retain(10f64.powf(decimals_0 - decimals_1)).unwrap());
    return price.unwrap();


    // public static fromX64(num: BN): Decimal {
    //     return new Decimal(num.toString()).mul(Decimal.pow(2, -64));
    //   }

    // public static sqrtPriceX64ToPrice(
    //     sqrtPriceX64: BN,
    //     decimalsA: number,
    //     decimalsB: number,
    //   ): Decimal {
    //     return MathUtil.fromX64(sqrtPriceX64)
    //       .pow(2)
    //       .mul(Decimal.pow(10, decimalsA - decimalsB));
    //   }
    
}