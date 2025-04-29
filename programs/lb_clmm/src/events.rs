use anchor_lang::event;
use anchor_lang::prelude::*;

#[event]
pub struct CompositionFee {
    // Sender's public key
    pub from: Pubkey,
    // Bin id
    pub bin_id: i16,
    // Amount of fee in token X
    pub token_x_fee_amount: u64,
    // Amount of fee in token Y
    pub token_y_fee_amount: u64,
    // Amount of protocol fee in token X
    pub protocol_token_x_fee_amount: u64,
    // Amount of protocol fee in token Y
    pub protocol_token_y_fee_amount: u64,
}

#[event]
pub struct AddLiquidity {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Sender's public key
    pub from: Pubkey,
    // Address of the position
    pub position: Pubkey,
    // Amount of token X, and Y deposited
    pub amounts: [u64; 2],
    // Pair active bin during deposit
    pub active_bin_id: i32,
}

#[event]
pub struct RemoveLiquidity {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Sender's public key
    pub from: Pubkey,
    // Address of the position
    pub position: Pubkey,
    // Amount of token X, and Y withdrawn
    pub amounts: [u64; 2],
    // Pair active bin during withdrawal
    pub active_bin_id: i32,
}

#[event]
pub struct Swap {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Address initiated the swap
    pub from: Pubkey,
    // Initial active bin ID
    pub start_bin_id: i32,
    // Finalized active bin ID
    pub end_bin_id: i32,
    // In token amount
    pub amount_in: u64,
    // Out token amount
    pub amount_out: u64,
    // Direction of the swap
    pub swap_for_y: bool,
    // Include protocol fee
    pub fee: u64,
    // Part of fee
    pub protocol_fee: u64,
    // Fee bps
    pub fee_bps: u128,
    // Host fee
    pub host_fee: u64,
}

#[event]
pub struct ClaimReward {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Position address
    pub position: Pubkey,
    // Owner of the position
    pub owner: Pubkey,
    // Index of the farm reward the owner is claiming
    pub reward_index: u64,
    // Total amount of reward claimed
    pub total_reward: u64,
}

#[event]
pub struct FundReward {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Address of the funder
    pub funder: Pubkey,
    // Index of the farm reward being funded
    pub reward_index: u64,
    // Amount of farm reward funded
    pub amount: u64,
}

#[event]
pub struct InitializeReward {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Mint address of the farm reward
    pub reward_mint: Pubkey,
    // Address of the funder
    pub funder: Pubkey,
    // Index of the farm reward being initialized
    pub reward_index: u64,
    // Duration of the farm reward in seconds
    pub reward_duration: u64,
}

#[event]
pub struct UpdateRewardDuration {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Index of the farm reward being updated
    pub reward_index: u64,
    // Old farm reward duration
    pub old_reward_duration: u64,
    // New farm reward duration
    pub new_reward_duration: u64,
}

#[event]
pub struct UpdateRewardFunder {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Index of the farm reward being updated
    pub reward_index: u64,
    // Address of the old farm reward funder
    pub old_funder: Pubkey,
    // Address of the new farm reward funder
    pub new_funder: Pubkey,
}

#[event]
pub struct PositionClose {
    // Address of the position
    pub position: Pubkey,
    // Owner of the position
    pub owner: Pubkey,
}

#[event]
pub struct ClaimFee {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Address of the position
    pub position: Pubkey,
    // Owner of the position
    pub owner: Pubkey,
    // Fee amount in token X
    pub fee_x: u64,
    // Fee amount in token Y
    pub fee_y: u64,
}

#[event]
pub struct LbPairCreate {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Bin step
    pub bin_step: u16,
    // Address of token X
    pub token_x: Pubkey,
    // Address of token Y
    pub token_y: Pubkey,
}

#[event]
pub struct PositionCreate {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Address of the position
    pub position: Pubkey,
    // Owner of the position
    pub owner: Pubkey,
}

#[event]
pub struct FeeParameterUpdate {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Protocol share in BPS
    pub protocol_share: u16,
    // Base factor of base fee rate
    pub base_factor: u16,
}

#[event]
pub struct IncreaseObservation {
    // Oracle address
    pub oracle: Pubkey,
    // Oracle length
    pub new_observation_length: u64,
}

#[event]
pub struct WithdrawIneligibleReward {
    // Liquidity pool pair
    pub lb_pair: Pubkey,
    // Reward mint
    pub reward_mint: Pubkey,
    // Amount of ineligible reward withdrawn
    pub amount: u64,
}

#[event]
pub struct UpdatePositionOperator {
    // Position public key
    pub position: Pubkey,
    // Old operator
    pub old_operator: Pubkey,
    // New operator
    pub new_operator: Pubkey,
}

#[event]
pub struct UpdatePositionLockReleaseSlot {
    // Position public key
    pub position: Pubkey,
    // Current slot
    pub current_slot: u64,
    // New lock release slot
    pub new_lock_release_slot: u64,
    // Old lock release slot
    pub old_lock_release_slot: u64,
    // Sender public key
    pub sender: Pubkey,
}
