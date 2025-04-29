use anchor_lang::prelude::*;
use solana_program::pubkey::Pubkey;

use super::lb_pair::{LbPair, PairStatus};

pub trait LbPairTypeActionAccess {
    fn validate_add_liquidity_access(&self, wallet: Pubkey) -> bool;
    fn validate_initialize_bin_array_access(&self, wallet: Pubkey) -> bool;
    fn validate_initialize_position_access(&self, wallet: Pubkey) -> bool;
    fn validate_swap_access(&self, sender: Pubkey) -> bool;
}

struct PermissionLbPairActionAccess {
    is_enabled: bool,
    activated: bool,
    pre_swap_activated: bool,
    whitelisted_wallet: Pubkey,
    pre_activation_swap_address: Pubkey,
}

impl PermissionLbPairActionAccess {
    pub fn new(lb_pair: &LbPair, current_slot: u64, pre_activation_swap_start_slot: u64) -> Self {
        Self {
            whitelisted_wallet: lb_pair.whitelisted_wallet,
            is_enabled: lb_pair.status == Into::<u8>::into(PairStatus::Enabled),
            activated: current_slot >= lb_pair.activation_slot,
            pre_activation_swap_address: lb_pair.pre_activation_swap_address,
            pre_swap_activated: current_slot >= pre_activation_swap_start_slot,
        }
    }
}

impl LbPairTypeActionAccess for PermissionLbPairActionAccess {
    fn validate_add_liquidity_access(&self, wallet: Pubkey) -> bool {
        // Pair disabled due to emergency mode. Nothing can be deposited.
        if !self.is_enabled {
            return false;
        }

        self.activated
            || !self.whitelisted_wallet.eq(&Pubkey::default())
                && self.whitelisted_wallet.eq(&wallet)
    }

    fn validate_initialize_bin_array_access(&self, wallet: Pubkey) -> bool {
        self.validate_add_liquidity_access(wallet)
    }

    fn validate_initialize_position_access(&self, wallet: Pubkey) -> bool {
        self.validate_add_liquidity_access(wallet)
    }

    fn validate_swap_access(&self, sender: Pubkey) -> bool {
        let activated = if self.pre_activation_swap_address.eq(&sender) {
            self.pre_swap_activated
        } else {
            self.activated
        };

        self.is_enabled && activated
    }
}

struct PermissionlessLbPairActionAccess {
    is_enabled: bool,
}

impl PermissionlessLbPairActionAccess {
    pub fn new(lb_pair: &LbPair) -> Self {
        Self {
            is_enabled: lb_pair.status == Into::<u8>::into(PairStatus::Enabled),
        }
    }
}

impl LbPairTypeActionAccess for PermissionlessLbPairActionAccess {
    fn validate_add_liquidity_access(&self, _wallet: Pubkey) -> bool {
        self.is_enabled
    }

    fn validate_initialize_bin_array_access(&self, _wallet: Pubkey) -> bool {
        self.is_enabled
    }

    fn validate_initialize_position_access(&self, _wallet: Pubkey) -> bool {
        self.is_enabled
    }

    fn validate_swap_access(&self, _sender: Pubkey) -> bool {
        self.is_enabled
    }
}

pub fn get_lb_pair_type_access_validator<'a>(
    lb_pair: &'a LbPair,
    current_slot: u64,
) -> Result<Box<dyn LbPairTypeActionAccess + 'a>> {
    if lb_pair.is_permission_pair()? {
        let pre_activation_start_slot = lb_pair.get_pre_activation_start_slot();
        let permission_pair_access_validator =
            PermissionLbPairActionAccess::new(lb_pair, current_slot, pre_activation_start_slot);

        Ok(Box::new(permission_pair_access_validator))
    } else {
        let permissionless_pair_access_validator = PermissionlessLbPairActionAccess::new(lb_pair);
        Ok(Box::new(permissionless_pair_access_validator))
    }
}
