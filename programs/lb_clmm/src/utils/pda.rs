use super::seeds::{self, BIN_ARRAY, BIN_ARRAY_BITMAP_SEED, ORACLE, PRESET_PARAMETER};
use anchor_lang::prelude::Pubkey;
use num_traits::ToBytes;
use std::{cmp::max, cmp::min};

pub fn derive_lb_pair_pda2(
    token_x_mint: Pubkey,
    token_y_mint: Pubkey,
    bin_step: u16,
    base_factor: u16,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            min(token_x_mint, token_y_mint).as_ref(),
            max(token_x_mint, token_y_mint).as_ref(),
            &bin_step.to_le_bytes(),
            &base_factor.to_le_bytes(),
        ],
        &crate::ID,
    )
}

pub fn derive_permission_lb_pair_pda(
    base: Pubkey,
    token_x_mint: Pubkey,
    token_y_mint: Pubkey,
    bin_step: u16,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            base.as_ref(),
            min(token_x_mint, token_y_mint).as_ref(),
            max(token_x_mint, token_y_mint).as_ref(),
            &bin_step.to_le_bytes(),
        ],
        &crate::ID,
    )
}

#[deprecated]
pub fn derive_lb_pair_pda(
    token_x_mint: Pubkey,
    token_y_mint: Pubkey,
    bin_step: u16,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            min(token_x_mint, token_y_mint).as_ref(),
            max(token_x_mint, token_y_mint).as_ref(),
            &bin_step.to_le_bytes(),
        ],
        &crate::ID,
    )
}

pub fn derive_position_pda(
    lb_pair: Pubkey,
    base: Pubkey,
    lower_bin_id: i32,
    width: i32,
) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            seeds::POSITION.as_ref(),
            lb_pair.as_ref(),
            base.as_ref(),
            lower_bin_id.to_le_bytes().as_ref(),
            width.to_le_bytes().as_ref(),
        ],
        &crate::ID,
    )
}

pub fn derive_oracle_pda(lb_pair: Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[ORACLE, lb_pair.as_ref()], &crate::ID)
}

pub fn derive_bin_array_pda(lb_pair: Pubkey, bin_array_index: i64) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[BIN_ARRAY, lb_pair.as_ref(), &bin_array_index.to_le_bytes()],
        &crate::ID,
    )
}

pub fn derive_bin_array_bitmap_extension(lb_pair: Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[BIN_ARRAY_BITMAP_SEED, lb_pair.as_ref()], &crate::ID)
}

pub fn derive_reserve_pda(token_mint: Pubkey, lb_pair: Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[lb_pair.as_ref(), token_mint.as_ref()], &crate::ID)
}

pub fn derive_reward_vault_pda(lb_pair: Pubkey, reward_index: u64) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[lb_pair.as_ref(), reward_index.to_le_bytes().as_ref()],
        &crate::ID,
    )
}

pub fn derive_event_authority_pda() -> (Pubkey, u8) {
    Pubkey::find_program_address(&[b"__event_authority"], &crate::ID)
}

#[deprecated]
pub fn derive_preset_parameter_pda(bin_step: u16) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[PRESET_PARAMETER, &bin_step.to_le_bytes()], &crate::ID)
}

pub fn derive_preset_parameter_pda2(bin_step: u16, base_factor: u16) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[
            PRESET_PARAMETER,
            &bin_step.to_le_bytes(),
            &base_factor.to_le_bytes(),
        ],
        &crate::ID,
    )
}
