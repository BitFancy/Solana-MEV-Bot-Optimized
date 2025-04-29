use crate::{assert_eq_launch_pool_admin, state::position::PositionV2};
use anchor_lang::prelude::*;

pub fn authorize_modify_position<'info>(
    position: &AccountLoader<'info, PositionV2>,
    sender: Pubkey,
) -> Result<bool> {
    let position = position.load()?;
    return Ok(position.owner == sender || position.operator == sender);
}

pub fn authorize_claim_fee_position<'info>(
    position: &AccountLoader<'info, PositionV2>,
    sender: Pubkey,
) -> Result<bool> {
    let position = position.load()?;

    if position.fee_owner == Pubkey::default() {
        Ok(position.owner == sender || position.operator == sender)
    } else {
        Ok(position.owner == sender
            || position.operator == sender
            || position.fee_owner == sender
            || assert_eq_launch_pool_admin(sender))
    }
}

pub trait PositionLiquidityFlowValidator {
    fn validate_outflow_to_ata_of_position_owner(&self, owner: Pubkey) -> Result<()>;
}
