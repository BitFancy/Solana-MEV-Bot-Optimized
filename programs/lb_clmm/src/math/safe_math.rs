use crate::errors::LBError;
use anchor_lang::solana_program::msg;
use ruint::aliases::U256;
use std::panic::Location;

pub trait SafeMath<T>: Sized {
    fn safe_add(self, rhs: Self) -> Result<Self, LBError>;
    fn safe_mul(self, rhs: Self) -> Result<Self, LBError>;
    fn safe_div(self, rhs: Self) -> Result<Self, LBError>;
    fn safe_rem(self, rhs: Self) -> Result<Self, LBError>;
    fn safe_sub(self, rhs: Self) -> Result<Self, LBError>;
    fn safe_shl(self, offset: T) -> Result<Self, LBError>;
    fn safe_shr(self, offset: T) -> Result<Self, LBError>;
}

macro_rules! checked_impl {
    ($t:ty, $offset:ty) => {
        impl SafeMath<$offset> for $t {
            #[inline(always)]
            fn safe_add(self, v: $t) -> Result<$t, LBError> {
                match self.checked_add(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }

            #[inline(always)]
            fn safe_sub(self, v: $t) -> Result<$t, LBError> {
                match self.checked_sub(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }

            #[inline(always)]
            fn safe_mul(self, v: $t) -> Result<$t, LBError> {
                match self.checked_mul(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }

            #[inline(always)]
            fn safe_div(self, v: $t) -> Result<$t, LBError> {
                match self.checked_div(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }

            #[inline(always)]
            fn safe_rem(self, v: $t) -> Result<$t, LBError> {
                match self.checked_rem(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }

            #[inline(always)]
            fn safe_shl(self, v: $offset) -> Result<$t, LBError> {
                match self.checked_shl(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }

            #[inline(always)]
            fn safe_shr(self, v: $offset) -> Result<$t, LBError> {
                match self.checked_shr(v) {
                    Some(result) => Ok(result),
                    None => {
                        let caller = Location::caller();
                        msg!("Math error thrown at {}:{}", caller.file(), caller.line());
                        Err(LBError::MathOverflow)
                    }
                }
            }
        }
    };
}

checked_impl!(u16, u32);
checked_impl!(i32, u32);
checked_impl!(u32, u32);
checked_impl!(u64, u32);
checked_impl!(i64, u32);
checked_impl!(u128, u32);
checked_impl!(i128, u32);
checked_impl!(usize, u32);
checked_impl!(U256, usize);
