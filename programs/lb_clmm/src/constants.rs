use anchor_lang::prelude::*;
use anchor_lang::solana_program::{pubkey, pubkey::Pubkey};

// TODO: Macro to compute the constants which changes based on the bit system used ?
// Smallest step between bin is 0.01%, 1 bps
#[constant]
pub const BASIS_POINT_MAX: i32 = 10000;

/// Maximum number of bin a bin array able to contains.
#[constant]
pub const MAX_BIN_PER_ARRAY: usize = 70;

/// Maximum number of bin per position contains.
#[constant]
pub const MAX_BIN_PER_POSITION: usize = 70;

/// Minimum bin ID supported. Computed based on 1 bps.
#[constant]
pub const MIN_BIN_ID: i32 = -443636;

/// Maximum bin ID supported. Computed based on 1 bps.
#[constant]
pub const MAX_BIN_ID: i32 = 443636;

/// Maximum fee rate. 10%
#[constant]
pub const MAX_FEE_RATE: u64 = 100_000_000;

#[constant]
pub const FEE_PRECISION: u64 = 1_000_000_000;

/// Maximum protocol share of the fee. 25%
#[constant]
pub const MAX_PROTOCOL_SHARE: u16 = 2_500;

/// Host fee. 20%
#[constant]
pub const HOST_FEE_BPS: u16 = 2_000;

pub const U24_MAX: u32 = 0xffffff;

// Number of rewards supported by pool
#[constant]
pub const NUM_REWARDS: usize = 2;

// Minimum reward duration
#[constant]
pub const MIN_REWARD_DURATION: u64 = 1;

#[constant]
pub const MAX_REWARD_DURATION: u64 = 31536000; // 1 year = 365 * 24 * 3600

pub const DEFAULT_OBSERVATION_LENGTH: u64 = 100;
pub const SAMPLE_LIFETIME: u64 = 120; // 2
#[constant]
pub const EXTENSION_BINARRAY_BITMAP_SIZE: usize = 12;

#[constant]
pub const BIN_ARRAY_BITMAP_SIZE: i32 = 512;

pub const MAX_BASE_FACTOR_STEP: u16 = 100; // 100 bps, 1%

pub const MAX_FEE_UPDATE_WINDOW: i64 = 0;

#[constant]
pub const MAX_REWARD_BIN_SPLIT: usize = 15;

#[cfg(feature = "localnet")]
pub static ALPHA_ACCESS_COLLECTION_MINTS: [Pubkey; 1] =
    [pubkey!("J1S9H3QjnRtBbbuD4HjPV6RpRhwuk4zKbxsnCHuTgh9w")];

#[cfg(not(feature = "localnet"))]
pub static ALPHA_ACCESS_COLLECTION_MINTS: [Pubkey; 1] =
    [pubkey!("5rwhXUgAAdbVEaFQzAwgrcWwoCqYGzR1Mo2KwUYfbRuS")];
